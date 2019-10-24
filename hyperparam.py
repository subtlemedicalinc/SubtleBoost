import numpy as np
from subtle.dnn.generators import GeneratorUNet2D
from subtle.data_loaders import SliceLoader
import subtle.subtle_loss as suloss

from test_tube import Experiment, HyperOptArgumentParser

fpaths_h5 = [
    '/home/srivathsa/projects/studies/gad/tiantan/preprocess/data/NO26.h5',
    '/home/srivathsa/projects/studies/gad/tiantan/preprocess/data/NO27.h5'
]

def train(params, gpu_ids):
    exp = Experiment(
        name=params.test_tube_exp_name,
        save_dir='/home/srivathsa/projects/studies/gad/tiantan/train/logs',
        autosave=False
    )
    exp.argparse(params)


    l1_w = params.l1_lambda
    ssim_w = 1 - l1_w
    loss_function = suloss.mixed_loss(l1_lambda=l1_w, ssim_lambda=ssim_w)
    metrics_monitor = [suloss.l1_loss, suloss.ssim_loss, suloss.mse_loss]

    data_loader = SliceLoader(
        data_list=fpaths_h5, batch_size=params.batch_size, shuffle=False, verbose=0, slices_per_input=7, resize=240, slice_axis=[0]
    )

    model = GeneratorUNet2D(
        num_channel_output=1,
        loss_function=loss_function, metrics_monitor=metrics_monitor,
        verbose=0, lr_init=params.lr_init,
        img_rows=240, img_cols=240, num_channel_input=14, compile_model=True
    )
    model.load_weights()

    train_X, train_Y = data_loader.__getitem__(7)
    val_X, val_Y = data_loader.__getitem__(14)

    for i in range(5):
        out = model.model.fit(train_X, train_Y, validation_data=(val_X, val_Y), verbose=0)
        print('history', out.history)
        val_l1_loss = np.array(out.history['val_l1_loss'])
        print('VAL LOSS', val_l1_loss)
        exp.log({'val_l1_loss': val_l1_loss})
    exp.save()
    exp.close()

if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='random_search')
    parser.add_argument('--test_tube_exp_name', default='test')
    parser.opt_range('--l1_lambda', default=0.6, type=float, tunable=True, low=0.0, high=1.0, nb_samples=20)
    parser.opt_list('--batch_size', default=8, type=int, tunable=True, options=[4, 8, 12, 16])
    parser.opt_range('--lr_init', default=0.001, type=float, tunable=True, low=0.1, high=0.001, nb_samples=20)

    hyp_params = parser.parse_args()
    hyp_params.optimize_parallel_gpu(train, gpu_ids=['0', '1'], max_nb_trials=5)
