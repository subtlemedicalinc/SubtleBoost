import numpy as np

MODEL_MAP = {
    'unet2d': {
        'model': 'generators.GeneratorUNet2D',
        'data_loader': 'slice_loader.SliceLoader'
    }
}

def make_image(im):
    """
    Convert an numpy representation image to Image protobuf.
    Copied from https://github.com/lanpa/tensorboard-pytorch/
    """
    import imageio
    import io
    nx, ny = im.shape
    im_uint = im.astype(np.uint8)
    output = io.BytesIO()
    imageio.imwrite(output, im, format='png')
    image_string = output.getvalue()
    output.close()
    return tf.Summary.Image(height=nx,
                         width=ny,
                         encoded_image_string=image_string)

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
