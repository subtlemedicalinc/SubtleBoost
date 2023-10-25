from keras.optimizers import Adam

from subtle.utils.experiment import get_model_config, get_layer_config

class AdversaryBase:
    def __init__(
        self, num_channel_input=1, img_rows=128, img_cols=128, num_filters_first_conv=32, batch_norm=True, verbose=True, compile_model=True, tunable_params=None, loss_function='mse', lr_init=2e-3, beta=0.5, model_config='base'
    ):
        self.num_channel_input = self.num_channel_output = num_channel_input
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.lr_init = lr_init
        self.beta = beta
        self.loss_function = loss_function
        self.verbose = verbose
        self.compile_model = compile_model
        self.model_config = model_config
        self.tunable_params = tunable_params

        self.model = None # to be assigned by _build_model() in children classes

        self._init_model_config()

    def get_config(self, param_name, layer_name=''):
        return get_layer_config(self.config_dict, param_name, layer_name)

    def _init_model_config(self):
        self.config_dict = get_model_config(self.model_name, self.model_config, model_type='adversaries', dirpath_config='./configs/models')

        if self.tunable_params:
            self.config_dict = {**self.config_dict, **self.tunable_params}

        for k, v in self.config_dict.items():
            if not isinstance(v, dict):
                # attributes like self.num_conv_per_pooling are assigned here
                setattr(self, k, v)

    def _compile_model(self):
        self.model.compile(
            loss=self.loss_function, optimizer=Adam(learning_rate=self.lr_init, beta_1=self.beta)
        )
