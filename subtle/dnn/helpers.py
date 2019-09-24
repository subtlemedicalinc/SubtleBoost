import numpy as np
import keras
from keras.layers import Input
import tensorflow as tf

MODEL_MAP = {
    'unet2d': 'generators.GeneratorUNet2D',
    'multires2d': 'generators.GeneratorMultiRes2D',
    'patch2d': 'adversaries.patch2d.AdversaryPatch2D'
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
    gen_img = gen(inputs)
    outputs = dis(gen_img)

    model = keras.models.Model(inputs=inputs, outputs=[gen_img, outputs])
    return model

def load_model(model_name):
    if model_name not in MODEL_MAP:
        raise ValueError('Model {} not supported'.format(model_name))

    mod_path = MODEL_MAP[model_name]
    mod_path = 'subtle.dnn.' + mod_path

    components = mod_path.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod