import numpy as np
import torch

MODEL_MAP = {
    'unet2d': {
        'model': 'generators.GeneratorUNet2D',
        'data_loader': 'slice_loader.SliceLoader'
    }
}

def make_image_grid(tensor_list):
    from torchvision.utils import make_grid
    images = []
    for tensor in tensor_list:
        image = tensor.unsqueeze(0)
        eps = 1e-7
        image = (image - image.min() + eps) / (image.max() - image.min() + eps)
        image = image.repeat(3, 1, 1)
        images.append(image)
    image_grid = make_grid(images, padding=0)
    return image_grid

def _load_module(mod_path):
    components = mod_path.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def load_model(args):
    if args.model_name not in MODEL_MAP:
        raise ValueError('Model {} not supported'.format(model_name))

    mod_path = MODEL_MAP[args.model_name]['model']
    mod_path = 'subtle.dnn.' + mod_path
    return _load_module(mod_path)

def load_db_class(args):
    if args.model_name not in MODEL_MAP:
        raise ValueError('Model {} not supported'.format(args.model_name))

    mod_path = MODEL_MAP[args.model_name]['data_loader']
    mod_path = 'subtle.data_loaders.' + mod_path

    return _load_module(mod_path)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if np.isfinite(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
