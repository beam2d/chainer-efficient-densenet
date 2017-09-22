import chainer
import fire


import densenet


def evaluate_densenet(n_layers, model_path, dataset_path, dataset_root='/', gpu=0):
    dn = getattr(densenet, 'DenseNetBC{}'.format(n_layers))()
    dn.load_caffe_model(model_path)
    model = chainer.links.Classifier(dn)
    if gpu >= 0:
        model.to_gpu(gpu)

    def crop(xt):
        x, t = xt
        y = x[:, 15:242, 15:242]
        return y, t

    dataset = chainer.datasets.TransformDataset(
        chainer.datasets.LabeledImageDataset(dataset_path, root=dataset_root),
        crop)
    iterator = chainer.iterators.SerialIterator(dataset, 100, repeat=False, shuffle=False)
    evaluator = chainer.training.extensions.Evaluator(iterator, model, device=gpu)
    result = evaluator()

    print(result)


if __name__ == '__main__':
    fire.Fire(evaluate_densenet)
