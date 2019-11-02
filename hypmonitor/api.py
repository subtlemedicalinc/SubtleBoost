import os
import numpy as np

import configargparse as argparse
import pandas as pd
from glob import glob

from flask import Flask, jsonify, request as req, render_template
app = Flask('hypmonitor', static_folder='./hypmonitor/static', template_folder='./hypmonitor/templates')

def create_app(hypsearch_path):
    app.config['hypsearch_path'] = hypsearch_path

def get_experiment_list(hypsearch_path):
    return [
        {'id': exp_path.split('/')[-1]}
        for exp_path in glob('{}/*'.format(hypsearch_path))
        if os.path.isdir(exp_path)
    ]

def get_trial_params(dirpath_trial):
    fpath_csv = os.path.join(dirpath_trial, 'params.csv')
    if not os.path.exists(fpath_csv) or os.stat(fpath_csv).st_size == 0:
        return None

    param_dict = pd.read_csv(fpath_csv).to_dict()

    return {v1: v2 for v1, v2 in zip(param_dict['Hyperparam'].values(), param_dict['Value'].values())}

def get_trial_metrics(dirpath_trial):
    fpath_csv = os.path.join(dirpath_trial, 'metrics.csv')
    if not os.path.exists(fpath_csv) or os.stat(fpath_csv).st_size == 0:
        return None

    metrics_dict = pd.read_csv(fpath_csv).T.to_dict()

    agg = [v['val_l1_loss'] for _, v in metrics_dict.items()]
    best_metrics = metrics_dict[np.argmin(agg)]

    return {
        'current_epoch': list(metrics_dict.keys())[-1],
        'best': {
            'epoch': best_metrics['epoch'],
            'val_l1_loss': best_metrics['val_l1_loss'],
            'val_ssim_loss': best_metrics['val_ssim_loss'],
            'val_psnr_loss': best_metrics['val_psnr_loss']
        }
    }

def get_epoch_metrics(dirpath_trial):
    metrics_dict = pd.read_csv(os.path.join(dirpath_trial, 'metrics.csv')).T.to_dict()

    return [v for _, v in metrics_dict.items()]

@app.route('/')
def index():
    return render_template('index.html', port=app.config.get('port'))

@app.route('/style.css')
def style():
    return app.send_static_file('style.css')

@app.route('/experiment')
def experiment():
    return render_template('experiment.html', id=req.args.get('id'), port=app.config.get('port'))

@app.route('/list')
def explist():
    return jsonify(get_experiment_list(app.config.get('hypsearch_path')))

@app.route('/logs')
def fetch_logs():
    pdicts = []
    dirpath_hyp = os.path.join(app.config.get('hypsearch_path'), req.args.get('experiment'))
    for dirpath_trial in glob('{}/trial_*'.format(dirpath_hyp)):
        trial_id = dirpath_trial.split('/')[-1]
        pdict = {'trial_id': trial_id}
        pdict['params'] = get_trial_params(dirpath_trial)
        pdict['metrics'] = get_trial_metrics(dirpath_trial)

        if pdict['params'] is not None and pdict['metrics'] is not None:
            pdicts.append(pdict)

    return jsonify(pdicts)

@app.route('/metrics')
def metrics():
    dirpath_trial = os.path.join(app.config.get('hypsearch_path'), req.args.get('experiment'), req.args.get('id'))
    return jsonify(get_epoch_metrics(dirpath_trial))

@app.route('/progress')
def progress():
    exp_id = req.args.get('experiment')
    trial_id = req.args.get('id')
    fpath_log = os.path.join(app.config.get('hypsearch_path'), exp_id, trial_id, 'log_train_tb_plot.log')
    fcontent = str(open(fpath_log, 'r').read())

    try:
        metrics_dict = pd.read_csv(os.path.join(os.path.dirname(fpath_log), 'metrics.csv')).T.to_dict()
    except Exception:
        metrics_dict = {}

    progress = []
    last_epoch = 0
    for key, val in metrics_dict.items():
        last_epoch = (key + 1)
        progress_item = 'End of epoch #{} - '.format(key + 1)
        progress_item += ' '.join(['{}: {:.3f}'.format(k, v) for k, v in val.items() if k != 'epoch'])
        progress.append(progress_item)

    last_progress = fcontent.split('Training')[-1]
    train_complete = ('done training' in last_progress)

    if 'done training' in last_progress:
        progress.append('done training')
    elif 'End of epoch' not in last_progress:
        progress.append('Training epoch #{} {}'.format(last_epoch + 1, last_progress))
    return jsonify({'train_log': progress, 'train_complete': train_complete})

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--hypsearch_path', type=str, action='store', default=None, help='Path from which to serve the hyperparam experiments results')
    parser.add_argument('--port', type=int, action='store', default=3333, help='Port on which to run the app')
    args = parser.parse_args()

    app.config['hypsearch_path'] = args.hypsearch_path
    app.config['port'] = args.port
    app.run(port=args.port)
