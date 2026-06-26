# Usage:       python test_pila_insar.py -r saved/okada_insar_A/<exp>/<run>/models/model_best.pth
# Description: Inference/analysis for PILA on InSAR line-of-sight (LOS) displacement.
#              Mirrors test_pila_mogi.py but output/target/residual columns are
#              LOS-per-point (one per downsampled observation point) instead of
#              ENU-per-station, and the inferred physics parameters (ATTRS) are
#              chosen by the physics type (Mogi_LOS or Okada_LOS).
# Date:        2026-06-11
import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from model import PHYS_VAE_SMPL  # PILA model
from parse_config import ConfigParser
import pandas as pd
import numpy as np

import os
import logging
from datetime import datetime
from pathlib import Path
from utils import read_json
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Inferred physics-parameter names by physics type (column order must match the
# enumeration order used by each decoder's rescale()).
PHYSICS_ATTRS = {
    'Mogi': ['xcen', 'ycen', 'd', 'dV'],
    'Mogi_LOS': ['xcen', 'ycen', 'd', 'dV'],
    'Okada': ['xoff', 'yoff', 'depth', 'strike', 'dip', 'length', 'width', 'opening'],
    'Okada_LOS': ['xoff', 'yoff', 'depth', 'strike', 'dip', 'length', 'width', 'opening'],
    'Sun69': ['xcen', 'ycen', 'depth', 'radius', 'dV'],
    'Sun69_LOS': ['xcen', 'ycen', 'depth', 'radius', 'dV'],
}


def setup_test_logging(checkpoint_path):
    """
    Setup logging to save test logs in the checkpoint's directory.
    """
    checkpoint_path = Path(checkpoint_path)
    # checkpoint is in models/ subdirectory, so go up one level for the experiment dir
    experiment_dir = checkpoint_path.parent.parent
    log_dir = experiment_dir / 'log'
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    log_filename = f'info_test_{timestamp}.log'
    log_file_path = log_dir / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler()  # Also log to console
        ]
    )

    logger = logging.getLogger('test')
    logger.info(f"Test logging setup. Log file: {log_file_path}")
    return logger


def main(config, args: argparse.Namespace):
    # Setup custom logging for test
    logger = setup_test_logging(args.resume)

    # setup data_loader instances
    data_loader = getattr(module_data, config.config['data_loader']['type_test'])(
        data_dir=config.config['data_loader']['data_dir_test'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        num_workers=0,
        with_const=config.config['data_loader']['args']['with_const'] if 'with_const' in config.config['data_loader']['args'] else False
    )

    # build model architecture
    model = PHYS_VAE_SMPL(config.config)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config.config['loss_test'])
    metric_fns = [getattr(module_metric, met) for met in config.config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(os.path.join(CURRENT_DIR, config.resume))
    state_dict = checkpoint['state_dict']
    if config.config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # Get config values first
    no_phy = config.config['arch']['phys_vae']['no_phy']
    dim_z_aux = config.config['arch']['phys_vae']['dim_z_aux']

    # Load tau/r values for inference if available
    tau_r_values = checkpoint.get('tau_r_values', None)
    if tau_r_values is not None and not no_phy:
        model.dec.set_tau_r_from_checkpoint(tau_r_values)
        logger.info(f"Loaded tau/r values from checkpoint: tau={tau_r_values['tau']:.3f}, r={tau_r_values['r']:.3f}")
    else:
        logger.info("Using default tau/r values for inference")

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    data_key = config.config['trainer']['input_key']
    target_key = config.config['trainer']['output_key']
    if 'input_const_keys' in config.config['trainer']:
        input_const_keys = config.config['trainer']['input_const_keys']
    else:
        input_const_keys = None
    no_phy = config.config['arch']['phys_vae']['no_phy']
    dim_z_aux = config.config['arch']['phys_vae']['dim_z_aux']

    # Inferred physics-parameter names depend on the physics type.
    physics = config.config['arch']['args'].get('physics', 'Okada_LOS')
    if not no_phy:
        ATTRS = PHYSICS_ATTRS[physics]
    else:
        ATTRS = [str(i + 1) for i in range(dim_z_aux)]

    # Load InSAR points info for LOS-per-point displacement columns. This is
    # required to label the output/target columns, so a load failure is fatal
    # (a silent empty dict would write a CSV with no displacement columns).
    points_path = os.path.join(CURRENT_DIR, config.config['arch']['args']['points_info'])
    if not os.path.exists(points_path):
        raise FileNotFoundError(f"points_info JSON not found: {points_path}")
    points_info = read_json(points_path)

    # One LOS observation per point (matches the N-dim decoder output ordering).
    POINTS = [f'los_{pid}' for pid in points_info.keys()]

    analyzer = {}

    with torch.no_grad():
        for batch_idx, data_dict in enumerate(data_loader):
            data = data_dict[data_key].to(device)
            target = data_dict[target_key].to(device)
            if input_const_keys is not None:
                input_const = {k: data_dict[k].to(device) for k in input_const_keys}
            else:
                input_const = None

            # Get time features if available
            time_feats = data_dict.get('time_feats', None)
            if time_feats is not None:
                time_feats = time_feats.to(device)
                if time_feats.dim() == 3:  # For sequence data
                    time_feats = time_feats.view(-1, time_feats.size(-1))

            if data.dim() == 3:
                sequence_len = data.size(1)
                data = data.view(-1, data.size(-1))
            if target.dim() == 3:
                target = target.view(-1, target.size(-1))

            # Determine hard_z based on whether KL terms were used during training
            if 'use_kl_term' in config.config['trainer']['phys_vae']:
                use_kl_term_old = config.config['trainer']['phys_vae']['use_kl_term']
                use_kl_term_z_phy = use_kl_term_old
                use_kl_term_z_aux = use_kl_term_old
            else:
                use_kl_term_z_phy = config.config['trainer']['phys_vae'].get('use_kl_term_z_phy', False)
                use_kl_term_z_aux = config.config['trainer']['phys_vae'].get('use_kl_term_z_aux', False)

            hard_z_phy = not use_kl_term_z_phy  # deterministic when KL term was disabled
            hard_z_aux = not use_kl_term_z_aux

            # Model returns: z_phy, z_aux, x_PB (physics+bias), x_P (raw physics)
            latent_phy, latent_aux, x_PB, x_P = model(
                data, t=time_feats, inference=True,
                hard_z_phy=hard_z_phy, hard_z_aux=hard_z_aux, const=input_const)

            if not no_phy:
                latent_phy = model.physics_model.rescale(latent_phy)
                latent = torch.stack([latent_phy[k] for k in latent_phy.keys()], dim=1)
                if dim_z_aux > 0:  # residual components present
                    bias = x_PB - x_P  # corrected output - raw physics output
                    data_concat(analyzer, 'init_output', x_P)
                    data_concat(analyzer, 'bias', bias)
                    data_concat(analyzer, 'latent_aux', latent_aux)
                else:
                    data_concat(analyzer, 'init_output', x_P)
                    data_concat(analyzer, 'bias', torch.zeros_like(x_P))
                    data_concat(analyzer, 'latent_aux', torch.zeros(x_P.shape[0], 0, device=x_P.device))
            else:
                latent = latent_aux

            data_concat(analyzer, 'output', x_PB)   # corrected output (physics + bias)
            data_concat(analyzer, 'target', target)
            data_concat(analyzer, 'latent', latent)
            if 'date' in data_dict:
                data_concat(analyzer, 'date', data_dict['date'])

            loss = loss_fn(x_PB, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(x_PB, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples
        for i, met in enumerate(metric_fns)
    })
    logger.info(log)

    # save the analyzer to csv using pandas
    columns = []
    columns += ['output' + '_' + b for b in POINTS]
    columns += ['target' + '_' + b for b in POINTS]
    columns += ['latent' + '_' + b for b in ATTRS]

    data = torch.hstack((
        analyzer['output'],
        analyzer['target'],
        analyzer['latent']
    ))

    if not no_phy:
        # Always include init_output and bias columns for consistency
        columns += ['init_output_' + b for b in POINTS]
        columns += ['bias_' + b for b in POINTS]
        data = torch.hstack((
            data,
            analyzer['init_output'],
            analyzer['bias']
        ))

        # Only include latent_aux columns when dim_z_aux > 0
        if dim_z_aux > 0:
            columns += ['latent_aux_' + str(b + 1) for b in range(dim_z_aux)]
            data = torch.hstack((
                data,
                analyzer['latent_aux']
            ))

    # Create a pandas dataframe
    data = data.cpu().numpy()
    df = pd.DataFrame(columns=columns, data=data)
    # Add date to the dataframe if it was collected
    if 'date' in analyzer:
        df['date'] = analyzer['date']

    out_path = os.path.join(CURRENT_DIR, str(config.resume).split('.pth')[0] + '_testset_analyzer.csv')
    df.to_csv(out_path, index=False)
    logger.info('Analyzer saved to {}'.format(out_path))


def data_concat(analyzer: dict, key: str, data):
    if key not in analyzer:
        analyzer[key] = data
    elif type(data) == torch.Tensor:
        analyzer[key] = torch.cat((analyzer[key], data), dim=0)
    elif type(data) == list:
        analyzer[key] = analyzer[key] + data


class TestConfigParser:
    """
    Custom config parser for testing that doesn't create new experiment directories.
    """
    def __init__(self, config_dict, resume_path):
        self._config = config_dict
        self.resume = resume_path

    @classmethod
    def from_args(cls, args):
        """Initialize from command line arguments"""
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Resume path need to be specified for testing."
            assert args.resume is not None, msg_no_cfg

        config = read_json(cfg_fname)
        return cls(config, args.resume)

    @property
    def config(self):
        return self._config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PILA InSAR inference')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()

    # Use custom config parser for testing
    config = TestConfigParser.from_args(args)
    main(config, args)
