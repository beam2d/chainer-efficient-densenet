import weakref

import chainer
from chainer import cuda
import chainer.functions as F
from chainer.functions.connection.convolution_2d import Convolution2DFunction
from chainer.functions.normalization.batch_normalization import BatchNormalization
import chainer.links as L
from chainer.links.caffe import caffe_function, CaffeFunction
try:
    import cupy
    fuse = cupy.fuse
except ImportError:
    fuse = lambda *_, **__: lambda f: f
import numpy


# Moncky patch to CaffeFunction: the Scale layers of pretrained DenseNet models have bias_term flag False, even though
# they actually have bias parameters. This patch forces CaffeFunction to load bias parameters even if bias_term is
# False.
@caffe_function._layer('Scale', None)
def _setup_scale(self, layer):
    bottom = layer.bottom
    blobs = layer.blobs
    axis = layer.scale_param.axis
    bias_term = True  # !!!PATCHED!!!

    # Case of only one bottom where W is learnt parameter.
    if len(bottom) == 1:
        W_shape = blobs[0].shape.dim
        func = L.Scale(axis, W_shape, bias_term)
        func.W.data.ravel()[:] = blobs[0].data
        if bias_term:
            func.bias.b.data.ravel()[:] = blobs[1].data
    # Case of two bottoms where W is given as a bottom.
    else:
        shape = blobs[0].shape.dim if bias_term else None
        func = L.Scale(
            axis, bias_term=bias_term, bias_shape=shape)
        if bias_term:
            func.bias.b.data.ravel()[:] = blobs[0].data

    # Add layer.
    with self.init_scope():
        setattr(self, layer.name, func)
    self.forwards[layer.name] = caffe_function._CallChildLink(self, layer.name)
    self._add_layer(layer)


CaffeFunction._setup_scale = _setup_scale


class InplaceConcatenator(object):

    """Large storage to concatenate feature maps in place."""

    def __init__(self, full_size):
        self._full_size = full_size
        self._storage = None
        self._sizes = []

    def forward(self, to, x):
        return InplaceConcat(self)(to, x)

    def alloc(self, x):
        xp = cuda.get_array_module(x.array)
        shape = list(x.shape)
        shape[1] = self._full_size
        self._storage = xp.empty(shape, dtype=x.dtype)

        new_array = self._storage[:, :x.shape[1]]
        new_array[...] = x.array
        x.array = new_array

    def concat(self, to, x):
        assert to.base is self._storage
        old_channels = to.shape[1]
        new_channels = old_channels + x.shape[1]
        new = self._storage[:, :new_channels]
        new[:, old_channels:] = x
        return new


class InplaceConcat(chainer.Function):

    def __init__(self, concatenator):
        self._concatenator = concatenator

    def forward(self, inputs):
        self.retain_inputs(())
        return self._concatenator.concat(*inputs),

    def backward(self, inputs, grad_outputs):
        if len(self.inputs) == 1:
            return grad_outputs
        gy, = grad_outputs
        n_channels_first = self.inputs[0].shape[1]
        return gy[:, :n_channels_first], gy[:, n_channels_first:]


def _load_caffe_bn(layers, prefix):
    bn = layers[prefix + '/bn']
    scale = layers[prefix + '/scale']
    with bn.init_scope():
        bn.gamma = scale.W
        bn.beta = scale.bias.b
    bn.eps = max(bn.eps, 1e-5)  # the pretrained model has eps slightly less than 1e-5 which is not allowed by cuDNN
    return bn


class RecomputedBNReluConv(chainer.Chain):

    def __init__(self, in_channels, out_channels, kernel_size, pad):
        super(RecomputedBNReluConv, self).__init__()
        self.pad = pad
        with self.init_scope():
            self.bn = L.BatchNormalization(in_channels)
            self.conv = L.Convolution2D(
                in_channels, out_channels, kernel_size, 1, pad, initialW=chainer.initializers.HeNormal(), nobias=True)

    def __call__(self, x):
        bn_fn = None
        conv_fn = None
        out_size = None

        def forward(x):
            nonlocal bn_fn, conv_fn, out_size

            if not chainer.config.enable_backprop:
                # forget phase
                with chainer.force_backprop_mode():
                    y = self.bn(x)
                bn_fn = y.creator
                bn_fn.unchain()

                y = F.relu(y)

                with chainer.force_backprop_mode():
                    y = self.conv(y)
                conv_fn = y.creator
                conv_fn.unchain()

                out_size = y.shape
                return y

            # recompute bn using computed statistics
            expander = bn_fn.expander
            bn_out = self._recompute_bn(x.array, self.bn.gamma.array[expander], self.bn.beta.array[expander],
                                        bn_fn.mean[expander], bn_fn.inv_std[expander])
            bn_out = chainer.Variable(bn_out)
            bn_fn.inputs = x.node, self.bn.gamma.node, self.bn.beta.node
            bn_fn.outputs = weakref.ref(bn_out.node),
            bn_out.creator_node = bn_fn
            x.retain_data()
            self.bn.gamma.retain_data()
            self.bn.beta.retain_data()

            # recompute relu
            h = F.relu(bn_out)

            # set dummy data to convolution output
            xp = cuda.get_array_module(h.array)
            conv_fn.inputs = h.node, self.conv.W.node
            h.retain_data()
            self.conv.W.retain_data()
            dummy_out = chainer.Variable(xp.broadcast_to(xp.empty((), dtype=h.dtype), out_size))
            conv_fn.outputs = weakref.ref(dummy_out.node),
            dummy_out.creator_node = conv_fn

            bn_fn = None
            conv_fn = None
            return dummy_out

        return F.forget(forward, x)

    def load_caffe_layers(self, layers, prefix):
        self.conv = layers[prefix]
        self.bn = _load_caffe_bn(layers, prefix)

    @staticmethod
    @fuse()
    def _recompute_bn(x, gamma, beta, mean, inv_std):
        return (x - mean) * inv_std * gamma + beta


class DenseLayer(chainer.Chain):

    def __init__(self, concatenator, in_ch, growth_rate):
        super(DenseLayer, self).__init__()
        initW = chainer.initializers.HeNormal()
        mid_ch = growth_rate * 4
        self.concatenator = concatenator

        with self.init_scope():
            self.l1 = RecomputedBNReluConv(in_ch, mid_ch, 1, 0)
            self.l2 = RecomputedBNReluConv(mid_ch, growth_rate, 3, 1)

    def __call__(self, x):
        h = self.l1(x)
        h = self.l2(h)
        return self.concatenator.forward(x, h)

    def load_caffe_layers(self, layers, prefix):
        self.l1.load_caffe_layers(layers, prefix + '/x1')
        self.l2.load_caffe_layers(layers, prefix + '/x2')


class TransitionLayer(chainer.Chain):

    def __init__(self, in_ch, out_ch):
        super(TransitionLayer, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(in_ch)
            self.conv = L.Convolution2D(in_ch, out_ch, 1, 1, 0)

    def __call__(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = F.average_pooling_2d(x, 2, stride=2)
        return x

    def load_caffe_layers(self, layers, prefix):
        self.bn = _load_caffe_bn(layers, prefix)
        self.conv = layers[prefix]


class DenseBlock(chainer.Chain):

    def __init__(self, num_layers, in_ch, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = []
        self.concatenator = InplaceConcatenator(in_ch + num_layers * growth_rate)

        ch = in_ch
        with self.init_scope():
            for i in range(num_layers):
                layer = DenseLayer(self.concatenator, ch, growth_rate)
                ch += growth_rate
                setattr(self, 'layer{}'.format(i), layer)
                self.layers.append(layer)

    def __call__(self, x):
        # assume x is a variable
        self.concatenator.alloc(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def load_caffe_layers(self, layers, prefix):
        for i, layer in enumerate(self.layers):
            layer.load_caffe_layers(layers, '{}_{}'.format(prefix, i + 1))


class DenseNetBC(chainer.Chain):

    def __init__(self, stage_sizes, growth_rate, reduction=0.5):
        super(DenseNetBC, self).__init__()
        ch = growth_rate * 2
        self.stages = []
        self.n_blocks = len(stage_sizes)

        with self.init_scope():
            self.conv1 = L.Convolution2D(
                3, ch, 7, 2, 3, initialW=chainer.initializers.HeNormal(), nobias=True)
            self.bn1 = L.BatchNormalization(ch)

            for i, s in enumerate(stage_sizes):
                block = DenseBlock(s, ch, growth_rate)
                ch += s * growth_rate
                setattr(self, 'block{}'.format(i), block)
                self.stages.append(block)

                if i + 1 < len(stage_sizes):
                    ch2 = int(ch * reduction)
                    trans = TransitionLayer(ch, ch2)
                    ch = ch2
                    setattr(self, 'trans{}'.format(i), trans)
                    self.stages.append(trans)

            self.fc_bn = L.BatchNormalization(ch)
            self.fc = L.Linear(ch, 1000)

    def __call__(self, x):
        xp = chainer.cuda.get_array_module(x, x.data)
        mean_bgr = xp.array([103.0626238, 115.90288257, 123.15163084])
        x -= mean_bgr[None, :, None, None]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pooling_2d(x, 3, stride=2)
        for stage in self.stages:
            x = stage(x)
        x = F.relu(self.fc_bn(x))
        x = F.average_pooling_2d(x, 7, stride=1)
        x = self.fc(x)
        return x

    def load_caffe_model(self, model_path):
        caffe = CaffeFunction(model_path)
        layers = dict(caffe.namedlinks())

        self.conv1 = layers['/conv1']
        self.bn1 = _load_caffe_bn(layers, '/conv1')

        for i in range(self.n_blocks):
            conv_name = '/conv{}'.format(i + 2)
            block = getattr(self, 'block{}'.format(i))
            block.load_caffe_layers(layers, conv_name)
            if i + 1 < self.n_blocks:
                trans = getattr(self, 'trans{}'.format(i))
                trans.load_caffe_layers(layers, conv_name + '_blk')

        self.fc_bn = _load_caffe_bn(layers, conv_name + '_blk')
        fc = layers['/fc{}'.format(self.n_blocks + 2)]  # conv
        self.fc.W.array = fc.W.array[:, :, 0, 0]
        self.fc.b.array = fc.b.array


class DenseNetBC121(DenseNetBC):
    
    def __init__(self):
        super(DenseNetBC121, self).__init__((6, 12, 24, 16), 32)


class DenseNetBC169(DenseNetBC):

    def __init__(self):
        super(DenseNetBC169, self).__init__((6, 12, 32, 32), 32)


class DenseNetBC201(DenseNetBC):

    def __init__(self):
        super(DenseNetBC201, self).__init__((6, 12, 48, 32), 32)


class DenseNetBC161(DenseNetBC):

    def __init__(self):
        super(DenseNetBC161, self).__init__((6, 12, 36, 24), 48)
