import numpy as np
from subtle.dnn.generators import GeneratorUNet2D
from subtle.data_loaders import SliceLoader
import subtle.subtle_loss as suloss

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
import keras
from ray.tune import track

ray.init()

fpaths_h5 = [
    '/home/srivathsa/projects/studies/gad/tiantan/preprocess/data/NO26.h5',
    '/home/srivathsa/projects/studies/gad/tiantan/preprocess/data/NO27.h5'
]

def train(params, reporter):
    l1_w = params['l1_lambda']
    ssim_w = 1 - l1_w
    loss_function = suloss.mixed_loss(l1_lambda=l1_w, ssim_lambda=ssim_w)
    metrics_monitor = [suloss.l1_loss, suloss.ssim_loss, suloss.mse_loss]

    data_loader = SliceLoader(
        data_list=fpaths_h5, batch_size=params['batch_size'], shuffle=False, verbose=0,
        slices_per_input=7, resize=240, slice_axis=[0]
    )

    model = GeneratorUNet2D(
        num_channel_output=1,
        loss_function=loss_function, metrics_monitor=metrics_monitor,
        verbose=0, lr_init=params['lr_init'],
        img_rows=240, img_cols=240, num_channel_input=14, compile_model=True
    )
    model.load_weights()

    train_X, train_Y = data_loader.__getitem__(7)
    val_X, val_Y = data_loader.__getitem__(14)

    return model.model.fit(train_X, train_Y, validation_data=(val_X, val_Y), verbose=0, callbacks=[TuneReporterCallback(reporter)])

class TuneReporterCallback(keras.callbacks.Callback):
    """Tune Callback for Keras."""

    def __init__(self, reporter=None, freq="batch", metrics=None, logs={}):
        """Initializer.
        Args:
            reporter (StatusReporter|tune.track.log|None): Tune object for
                returning results.
            freq (str): Sets the frequency of reporting intermediate results.
                One of ["batch", "epoch"].
        """
        self.reporter = reporter or track.log
        self.iteration = 0
        self.metrics = metrics or None
        if freq not in ["batch", "epoch"]:
            raise ValueError("{} not supported as a frequency.".format(freq))
        self.freq = freq
        super(TuneReporterCallback, self).__init__()

    def on_batch_end(self, batch, logs={}):
        if not self.freq == "batch":
            return
        self.iteration += 1
        # for metric in list(logs):
        #     if "loss" in metric and "neg_" not in metric:
        #         logs["neg_" + metric] = -logs[metric]
        print('logs....', logs)
        self.reporter(keras_info=logs, mean_accuracy=-logs.get('l1_loss'))

    def on_epoch_end(self, batch, logs={}):
        if not self.freq == "epoch":
            return
        self.iteration += 1
        # for metric in list(logs):
        #     if "loss" in metric and "neg_" not in metric:
        #         logs["neg_" + metric] = -logs[metric]
        self.reporter(keras_info=logs, mean_accuracy=-logs.get("val_l1_loss"))

if __name__ == '__main__':
    sched = AsyncHyperBandScheduler(
        metric="val_l1_loss",
        mode="min"
    )

    tune.run(
        train,
        name="exp",
        scheduler=sched,
        num_samples=2,
        config={
            "num_workers": 2,
            "num_gpus": 2,
            "num_gpus_per_worker": 1,
            "num_cpus": 0,
            "l1_lambda": tune.sample_from(lambda spec: np.random.uniform(0, 1)),
            "batch_size": tune.sample_from(lambda spec: [4, 8, 12][np.random.randint(3)]),
            "lr_init": tune.sample_from(lambda spec: np.random.uniform(0.001, 0.1))
        })
