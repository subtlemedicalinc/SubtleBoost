import numpy as np
import keras
from keras.layers import Input
import tensorflow as tf

MODEL_MAP = {
    'unet2d': {
        'model': 'generators.GeneratorUNet2D',
        'data_loader': 'slice_loader.SliceLoader'
    },
    'attn_unet2d': {
        'model': 'generators.GeneratorAttnUNet2D',
        'data_loader': 'slice_loader.SliceLoader'
    },
    'mres2d': {
        'model': 'generators.GeneratorMultiRes2D',
        'data_loader': 'slice_loader.SliceLoader'
    },
    'patch2d': {
        'model': 'adversaries.AdversaryPatch2D'
    },
    'dense2d': {
        'model': 'adversaries.AdversaryDense2D'
    },
    'unet3d': {
        'model': 'generators.GeneratorUNet3D',
        'data_loader': 'block_loader.BlockLoader'
    },
    'edsr2d': {
        'model': 'generators.GeneratorEDSR2D',
        'data_loader': 'slice_loader.SliceLoader'
    },
    'rrdb2d': {
        'model': 'generators.GeneratorRRDB2D',
        'data_loader': 'slice_loader.SliceLoader'
    },
    'edsr3d': {
        'model': 'generators.GeneratorEDSR3D',
        'data_loader': 'block_loader.BlockLoader'
    },
    'vdsr3d': {
        'model': 'generators.GeneratorVDSR3D',
        'data_loader': 'block_loader.BlockLoader'
    },
    'wdsr3d': {
        'model': 'generators.GeneratorWDSR3D',
        'data_loader': 'block_loader.BlockLoader'
    },
    'branch_unet2d': {
        'model': 'generators.GeneratorBranchUNet2D',
        'data_loader': 'slice_loader.SliceLoader'
    },
    'ivdnet2d': {
        'model': 'generators.GeneratorIVDNet2D',
        'data_loader': 'slice_loader.SliceLoader'
    },
    'fboost_unet2d': {
        'model': 'generators.GeneratorFBoostUNet2D',
        'data_loader': 'slice_loader.SliceLoader'
    },
    'series_unet2d': {
        'model': 'generators.GeneratorSeriesUNet2D',
        'data_loader': 'slice_loader.SliceLoader'
    }
}

# clean up
def clear_keras_memory():
    keras.backend.clear_session()

# use part of memory
def set_keras_memory(limit=0.9):
    from tensorflow import ConfigProto as tf_ConfigProto
    from tensorflow import Session as tf_Session
    from keras.backend.tensorflow_backend import set_session
    config = tf_ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = limit
    config.gpu_options.allow_growth = True
    set_session(tf_Session(config=config))

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
def gan_model(gen, dis, input_shape):
    inputs = Input(shape=input_shape)

    gen.name = "gen"
    dis.name = "dis"
    gen_img = gen(inputs)
    outputs = dis(gen_img)

    model = keras.models.Model(inputs=inputs, outputs=[gen_img, outputs])
    return model

def _load_module(mod_path):
    components = mod_path.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def load_model(model_name):
    if model_name not in MODEL_MAP:
        raise ValueError('Model {} not supported'.format(model_name))

    mod_path = MODEL_MAP[model_name]['model']
    mod_path = 'subtle.dnn.' + mod_path
    return _load_module(mod_path)

def load_data_loader(model_name):
    if model_name not in MODEL_MAP:
        raise ValueError('Model {} not supported'.format(model_name))

    mod_path = MODEL_MAP[model_name]['data_loader']
    mod_path = 'subtle.data_loaders.' + mod_path
    return _load_module(mod_path)
