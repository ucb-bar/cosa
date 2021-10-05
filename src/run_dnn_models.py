#!/usr/bin/env python3 
import argparse
import logging
import os
import pathlib

import utils
from cosa import run_timeloop
from cosa_input_objs import Prob

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # capture everything
# logger.disabled = True

_COSA_DIR = os.environ['COSA_DIR']


def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')

    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default='output_dir',
                        )
    parser.add_argument('-ap',
                        '--arch_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{_COSA_DIR}/configs/arch/simba.yaml',
                        )
    parser.add_argument('-mp',
                        '--mapspace_path',
                        type=str,
                        help='Mapspace Path',
                        default=f'{_COSA_DIR}/configs/mapspace/mapspace.yaml',
                        )
    parser.add_argument('-w',
                        '--workload_dir',
                        type=str,
                        help='DNN Model Workload Folder',
                        default=f'{_COSA_DIR}/configs/workloads',
                        )
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        help='DNN Model',
                        )
    return parser


def run_dnn_models(workload_dir, arch_path, mapspace_path, output_dir, model):
    cycle_result_path = pathlib.Path(output_dir) / f'results_{arch_path.stem}_cycle.json'
    if cycle_result_path.exists():
        return

    if model is not None:
        model_strs = [model]
    else:
        model_strs = ['alexnet', 'resnet50', 'resnext50_32x4d', 'deepbench']
    full_results = {'energy': {}, 'pe_cycle': {}, 'cycle': {}, 'area': {}}

    for model_str in model_strs:
        logger.info(f'model: {model_str}')
        for k in full_results.keys():
            full_results[k][model_str] = {}

        model_dir = workload_dir / (model_str + '_graph')
        layer_def_path = model_dir / 'unique_layers.yaml'
        layers = utils.parse_yaml(layer_def_path)

        # Schedule each layer
        for layer_idx, layer in enumerate(layers):
            prob_path = model_dir / (layer + '.yaml')
            status_dict = run_timeloop(prob_path, arch_path, mapspace_path, output_dir)
            status_dict_key = next(iter(status_dict))

            prob = Prob(prob_path)
            logger.info(f'prob_path: {prob_path}')
            try:
                for k in full_results:
                    full_results[k][model_str][prob.config_str()] = status_dict[status_dict_key][k]
            except:
                # Set to -1 and move on the next config
                full_results[k][model_str][prob.config_str()] = -1
                raise

    for k in full_results:
        full_result_path = pathlib.Path(output_dir) / f'results_{arch_path.stem}_{k}.json'
        utils.store_json(full_result_path, full_results[k], indent=2)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    workload_dir = pathlib.Path(args.workload_dir).resolve()
    arch_path = pathlib.Path(args.arch_path).resolve()
    mapspace_path = pathlib.Path(args.mapspace_path).resolve()
    output_dir = args.output_dir
    model = args.model

    run_dnn_models(workload_dir, arch_path, mapspace_path, output_dir, model)
