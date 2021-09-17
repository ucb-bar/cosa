#!/usr/bin/env python3 

import argparse
import logging
import os
import pathlib
import math

import utils
from cosa import run_timeloop
from cosa_input_objs import Prob
from run_arch import gen_arch_yaml_from_config, gen_data, fetch_data, gen_dataset 

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
                        default='output_dir_dse_os',
                        )
    parser.add_argument('-a',
                        '--arch_dir',
                        type=str,
                        help='Generated Archtecture Folder',
                        default='dse_arch_os',
                        )
    parser.add_argument('-c',
                        '--config_dir',
                        type=str,
                        help='DSE Config',
                        default='/scratch/qijing.huang/CoSA_VAE_DSE/results/cosa_data_VAE_predictor'
                        )
    parser.add_argument('-bap',
                        '--base_arch_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{_COSA_DIR}/configs/arch/simba.yaml',
                        )
    return parser


def round_config(i, base=2):
    return math.ceil(i / base) * base


def parse_hw_configs(config_dir, search_algo, opt_algo):
    search_dir = config_dir / search_algo
    glob_str = f'{opt_algo}_*.txt'
    config_files = list(search_dir.glob(glob_str))
    hw_configs = [] 
    for config_file in config_files: 
        with open(config_file, 'r') as f:
            lines = f.readlines()
        config_str = lines[0]
        config_str = config_str.replace(']','')
        config_str = config_str.replace('[','')
        confg_str_arr = config_str.split(' ')
        configs = [round_config(float(i)) for i in confg_str_arr]
        hw_configs.append(configs)
    print(hw_configs)
    return hw_configs


def eval(hw_config, arch_dir, output_dir, search_algo='', opt_algo='', arch_v3=False, unique_sum=True, workload_dir=None):
    hw_config = hw_config[:]
    for idx in range(len(hw_config)):
        orig = hw_config[idx]
        if idx == 0:
            new = pow(2, math.ceil(math.log(orig)/math.log(2)))
        else:
            new = round_config(orig)
        hw_config[idx] = new

    glob_str = f'arch_{search_algo}_{opt_algo}_*.yaml'
    config_yaml_str = gen_arch_yaml_from_config(base_arch_path, arch_dir, hw_config, search_algo, opt_algo, arch_v3=False)
    config_str = config_yaml_str.replace('.yaml', '')
    gen_data(arch_dir, output_dir, glob_str)
     
    cycle_path = output_dir / f'results_{config_str}_cycle.json'
    energy_path = output_dir / f'results_{config_str}_energy.json'
    area_path = output_dir /f'results_{config_str}_area.json'
    if unique_sum: 
        try:
            cycle = sum(utils.parse_json(cycle_path)['resnet50'].values())
            energy = sum(utils.parse_json(energy_path)['resnet50'].values())
            area = sum(utils.parse_json(area_path)['resnet50'].values())
        except:
            raise
    else:
        raise ValueError("Add non unique sum support!")
    return (cycle, energy, area)

    
def gen_results_dataset(base_arch_path, arch_dir, output_dir, config_dir):
    # search_algo = 'random_search'
    search_algo = 'optimal_search'
    opt_algo = 'sgd'
    glob_str = f'arch_{search_algo}_{opt_algo}_*.yaml'
    hw_configs = parse_hw_configs(config_dir, search_algo, opt_algo)
    for hw_config in hw_configs: 
        gen_arch_yaml_from_config(base_arch_path, arch_dir, hw_config, search_algo, opt_algo, arch_v3=False)

    gen_data(arch_dir, output_dir, glob_str)
    # fetch_data(arch_dir, output_dir, f"arch_{search_algo}_{opt_algo}_*.yaml")
    
    gen_dataset(arch_dir, output_dir, glob_str, arch_v3=False, mem_levels=5, model_cycles=False)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    base_arch_path = pathlib.Path(args.base_arch_path).resolve()
    arch_dir = pathlib.Path(args.arch_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    config_dir = pathlib.Path(args.config_dir).resolve()

    hw_config = [2,2,2,4,2,2,38874]
    eval_result = eval(hw_config, arch_dir, output_dir, search_algo='', opt_algo='', arch_v3=False, unique_sum=True, workload_dir=None)
    print(eval_result)

