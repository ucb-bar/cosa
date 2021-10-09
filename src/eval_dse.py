#!/usr/bin/env python3 

import argparse
import logging
import os
import pathlib
import math

import utils
from cosa import run_timeloop
from cosa_input_objs import Prob
from run_arch import gen_arch_yaml_from_config, gen_data, fetch_data, gen_dataset, parse_results 

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # capture everything
# logger.disabled = True

_COSA_DIR = os.environ['COSA_DIR']

def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('--obj', default='edp', help='valid options [edp, latency, energy]')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default='dse_output_dir_nz2_small_MN_predictor',
                        # default='dse_output_dir_deep_predictor_1',
                        )
    parser.add_argument('-a',
                        '--arch_dir',
                        type=str,
                        help='Generated Archtecture Folder',
                        default='dse_arch_nz2_small_MN_predictor',
                        )
    parser.add_argument('-c',
                        '--config_dir',
                        type=str,
                        help='DSE Config',
                        default='/scratch/qijing.huang/CoSA_VAE_DSE/results/cosa_data_VAE_nz2_small_MN_predictor'
                        # default='/scratch/qijing.huang/CoSA_VAE_DSE/results/cosa_data_VAE_nz2_predictor'
                        # default='/scratch/qijing.huang/CoSA_VAE_DSE/results/cosa_data_VAE_deep_pred_1_predictor'
                        # default='/scratch/qijing.huang/CoSA_VAE_DSE/results/cosa_data_VAE_new7_predictor'
                        # default='/scratch/qijing.huang/CoSA_VAE_DSE/results/cosa_data_VAE_predictor'
                        )
    parser.add_argument(
                        '--search_algo_postfix',
                        type=str,
                        help='Postfix for search algo for different seeds',
                        default='',
                        )

    parser.add_argument('-bap',
                        '--base_arch_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{_COSA_DIR}/configs/arch/simba_dse.yaml',
                        )
    parser.add_argument(
                        '--model',
                        type=str,
                        help='Target DNN Model',
                        default='resnet50',
                        )

    return parser


def round_config(i, base=2):
    return math.ceil(i / base) * base


def parse_hw_configs(config_dir, search_algo, opt_algo, search_algo_postfix=""):
    search_dir = config_dir / f'{search_algo}{search_algo_postfix}'
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
        configs = [pow(2, math.ceil(math.log(float(confg_str_arr[0]))/math.log(2)))] + [round_config(float(i)) for i in confg_str_arr[1:]]
        hw_configs.append(configs)
    print(hw_configs)
    return hw_configs


def total_layer_values(layer_values_dict, layer_count):
    """
    Calculate the total cycle/energy value of a network by summing up the values for each unique layer,
    multiplied by the number of times a layer with those dimensions appears in the network.
    """
    total = 0
    for layer in layer_values_dict:
        if layer not in layer_count:
            print(f"ERROR: layer {layer} not found in layer count file")
            exit(1)
        total += layer_values_dict[layer] * layer_count[layer]

    return total


def discretize_config(hw_config): 
    hw_config = hw_config[:]
    for idx in range(len(hw_config)):
        orig = hw_config[idx]
        if idx == 0:
            new = pow(2, math.ceil(math.log(orig)/math.log(2)))
        else:
            new = round_config(orig)
        hw_config[idx] = new
    return hw_config


def eval(hw_config, base_arch_path, arch_dir, output_dir, config_prefix='', arch_v3=False, unique_sum=True, workload_dir='../configs/workloads', model='resnet50', layer_idx=None):
    hw_config = discretize_config(hw_config)
    config_yaml_str = gen_arch_yaml_from_config(base_arch_path, arch_dir, hw_config, config_prefix, arch_v3=arch_v3)
    # glob_str = f'arch_{config_prefix}*.yaml'
    glob_str = config_yaml_str
    config_str = config_yaml_str.replace('.yaml', '')
    gen_data(arch_dir, output_dir, glob_str)
    cycle, energy, area = parse_results(output_dir, config_str, unique_sum, workload_dir=workload_dir, model=model, layer_idx=layer_idx)
    return (cycle, energy, area)

    
def gen_results_dataset(base_arch_path, arch_dir, output_dir, config_dir, search_algo_postfix='', model='resnet50', obj='edp'):
    search_algos = ['random_search', 'optimal_search']
    #search_algos = ['optimal_search']
    opt_algos = ['Newton', 'sgd']
    #opt_algos = ['Newton']
    best_perfs = []
    for search_algo in search_algos:
        for opt_algo in opt_algos: 
            glob_str = f'arch_{search_algo}_{opt_algo}_*.yaml'
            hw_configs = parse_hw_configs(config_dir, search_algo, opt_algo, search_algo_postfix)
            for hw_config in hw_configs: 
                config_prefix = f'{search_algo}_{opt_algo}_'
                gen_arch_yaml_from_config(base_arch_path, arch_dir, hw_config, config_prefix, arch_v3=False)

            gen_data(arch_dir, output_dir, glob_str, model=model)
            best_perf = gen_dataset(arch_dir, output_dir, glob_str=glob_str, model=model, arch_v3=False, mem_levels=5, model_cycles=False, postfix=f'_{search_algo}_{opt_algo}', obj=obj)
            best_perfs.append(best_perf)
    print("Optimal Design Points")
    print(best_perfs)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    base_arch_path = pathlib.Path(args.base_arch_path).resolve()
    arch_dir = pathlib.Path(args.arch_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    config_dir = pathlib.Path(args.config_dir).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    model = args.model
    
    gen_results_dataset(base_arch_path, arch_dir, output_dir, config_dir, search_algo_postfix=args.search_algo_postfix, model=model, obj=args.obj)
    
    #parse_search_results()
    
    # mesh x [3, 4, 5, 8, 10, 12, 14] 
    # 4,256,256,1,128,256,128,16384,64,4096,1,32768
    # hw_config = [4,256,256,16384,4096,32768]
    # eval_result = eval(hw_config, base_arch_path, arch_dir, output_dir, config_prefix='', arch_v3=False, unique_sum=True, workload_dir=None)
    # print(eval_result)
    # print(eval_result[0] * eval_result[1])
