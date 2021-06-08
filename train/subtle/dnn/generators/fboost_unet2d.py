import tensorflow as tf
import keras.models
import numpy as np

from keras.layers import Input, Lambda

from subtle.dnn.generators.base import GeneratorBase
from subtle.dnn.generators.unet2d import GeneratorUNet2D
from subtle.dnn.generators.branch_unet2d import GeneratorBranchUNet2D

class GeneratorFBoostUNet2D(GeneratorBase):
    def __init__(self, **kwargs):
        self.model_name = 'fboost_unet2d'
        super().__init__(**kwargs)
        self._build_model()

    def _transfer_weights(self, dest_model, branch_num, ckp_file, kw=None, op_layer=None, freeze=False, num_ip_channels=None):
        ip_ch = num_ip_channels if num_ip_channels is not None else self.submodel_num_channel
        src_model = GeneratorUNet2D(
            num_channel_input=ip_ch, num_channel_output=self.num_channel_output,
            img_rows=self.img_rows, img_cols=self.img_cols,
            verbose=self.verbose,
            compile_model=False,
            model_config='base',
            checkpoint_file=ckp_file
        )
        src_model.load_weights()

        if kw is None and op_layer is None:
            kw = 'b{}_'.format(branch_num)
            op_layer = 'branch{}_output'.format(branch_num)

        print('Transferring weights -> {} from {}'.format(kw, ckp_file))
        src_layers = [l.name for l in src_model.model.layers]
        for idx, layer in enumerate(dest_model.model.layers):
            if kw not in layer.name:
                continue
            unet_name = layer.name.replace(kw, '')

            if unet_name in src_layers:
                lname = unet_name
                lname_orig = layer.name
                src_weights = src_model.model.layers[src_layers.index(unet_name)].get_weights()
                dest_model.model.layers[idx].set_weights(src_weights)
                dest_model.model.layers[idx].trainable = (not freeze)

        dest_model.model.get_layer(op_layer).set_weights(
            src_model.model.get_layer('model_output').get_weights()
        )
        dest_model.model.get_layer(op_layer).trainable = (not freeze)
        return dest_model

    def _transfer_all_weights(self, model_main):
        model_main = self._transfer_weights(model_main, '1', self.fpaths_pre[0], freeze=True)
        model_main = self._transfer_weights(model_main, '2', self.fpaths_pre[1], freeze=True)
        model_main = self._transfer_weights(model_main, '3', self.fpaths_pre[2], freeze=True)
        model_main = self._transfer_weights(model_main, '4', self.fpaths_pre[3], freeze=True)

        if len(self.fpaths_pre) == 5:
            model_main = self._transfer_weights(model_main, 'fbst_', self.fpaths_pre[4], kw='fbst_', op_layer='model_output', freeze=False, num_ip_channels=2)
        return model_main


    def _build_model(self):
        print('Building {}-{} model...'.format(self.model_name, self.model_config))

        # layers
        # 2D input is (rows, cols, channels)

        inputs = Input(shape=(self.img_rows, self.img_cols, self.num_channel_input), name='model_input')

        print('inputs', inputs)
        num_mods = self.num_modalities

        ip_names = ['t1_pre', 't1_low', 't2', 'fl', 'uad']
        t1_pre, t1_low, t2, flair, uad = [
            Lambda(lambda ip: ip[..., idx::num_mods], name=ip_names[idx])(inputs)
            for idx in np.arange(num_mods)
        ]

        model_main = GeneratorBranchUNet2D(
            num_channel_input=self.num_channel_input, num_channel_output=self.num_channel_output,
            img_rows=self.img_rows, img_cols=self.img_cols,
            verbose=self.verbose,
            compile_model=False,
            model_config=self.branch_unet_mode
        )

        if self.transfer_weights:
            model_main = self._transfer_all_weights(model_main)

        if self.verbose:
            print(model_main)

        self.model = model_main.model
        self.model.summary()

        if self.compile_model:
            self._compile_model()
