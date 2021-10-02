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
    parser.add_argument('-bap',
                        '--base_arch_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{_COSA_DIR}/configs/arch/simba_dse.yaml',
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

    
def gen_results_dataset(base_arch_path, arch_dir, output_dir, config_dir):
    search_algos = ['random_search', 'optimal_search']
    #search_algos = ['optimal_search']
    opt_algos = ['Newton', 'sgd']
    #opt_algos = ['Newton']
    best_perfs = []
    for search_algo in search_algos:
        for opt_algo in opt_algos: 
            glob_str = f'arch_{search_algo}_{opt_algo}_*.yaml'
            hw_configs = parse_hw_configs(config_dir, search_algo, opt_algo)
            for hw_config in hw_configs: 
                config_prefix = f'{search_algo}_{opt_algo}_'
                gen_arch_yaml_from_config(base_arch_path, arch_dir, hw_config, config_prefix, arch_v3=False)

            gen_data(arch_dir, output_dir, glob_str)
            best_perf = gen_dataset(arch_dir, output_dir, glob_str, arch_v3=False, mem_levels=5, model_cycles=False, postfix=f'_{search_algo}_{opt_algo}')
            best_perfs.append(best_perf)
    print("Optimal Design Points")
    print(best_perfs)


def get_best_entry(data, metric_idx=[1,2]):
    best_perf = None
    best_entry = None
    for line, per_arch_data in enumerate(data): 
        perf_prod = 1.0
        for entry in metric_idx:
            perf_prod *= float(per_arch_data[entry])
        if line == 0:
            best_perf = perf_prod
            best_entry = per_arch_data
        if perf_prod < best_perf:
            best_perf = perf_prod
            best_entry = per_arch_data
    return best_perf, best_entry


def parse_best_results(dataset_path, n_entries=None):
    data = utils.parse_csv(dataset_path)
    if n_entries is None:
        data = data[1:]
    else:
        data = data[1: n_entries+1]
    best_metric, best_entry = get_best_entry(data) 
    print(f'dataset_path: {dataset_path}') 
    print(f'best_entry: {best_entry}') 
    print(f'best_metric: {best_metric}') 
    return best_metric, best_entry


def parse_search_results():
    samples = [50, 100, 500, 1000, 2000]
    result_data = {}
    result_data['num_sample'] = samples

    result_data['dataset_best_1'] = [8798868924440581.0, 8798868924440581.0, 8676699496627586.0, 8676699496627586.0, 8676699496627586.0] 
    result_data['dataset_mean_1'] = [3.321781171088311e+16, 3.0531516714279996e+16, 2.898710045164625e+16, 2.9947367986199504e+16, 2.999884242291844e+16] 
    result_data['dataset_median_1'] = [3.6414777147006296e+16, 3.2195480454620108e+16, 2.4187872490839704e+16, 2.880585407106034e+16, 2.7941558106960584e+16] 
    
    # parse original dataset
    path = pathlib.Path('/scratch/qijing.huang/cosa/src') 

    search_algos = ['random_search', 'optimal_search']
    opt_algos = ['Newton', 'sgd']

    # parse VAE results 
    seeds = [1, 1234]
    for seed in seeds: 
        for search_algo in search_algos: 
            for opt_algo in opt_algos: 
                entry_name = f'vae_{search_algo}_{opt_algo}_{seed}'
                if seed == 1:
                    seed_str=''
                else:
                    seed_str=f'_seed_{seed}'
                data_entry = []
                for num_sample in samples: 
                    output_dir = path / f'dse_output_dir_dataset_{num_sample}{seed_str}'  
                    dataset_path = output_dir / f'dataset_{search_algo}_{opt_algo}.csv'
                    best_metric, _ = parse_best_results(dataset_path, num_sample)
                    data_entry.append(best_metric)
                result_data[entry_name] = data_entry

    # parse BO results 
    seeds = [1, 1234]
    for seed in seeds:
        entry_name = f'bo_{seed}'
        data_entry = []
        for num_sample in samples:
            output_dir_bo = pathlib.Path('/nscratch/qijing.huang/cosa/results/bo_search/')
            dataset_path = output_dir_bo / f'dataset_s{seed}.csv'
            best_metric, _ = parse_best_results(dataset_path, num_sample)
            data_entry.append(best_metric)
        result_data[entry_name] = data_entry
            
    seeds = [6, 7]
    for seed in seeds:
        entry_name = f'random_{seed}'
        data_entry = []
        for num_sample in samples:
            output_dir_bo = pathlib.Path('/nscratch/qijing.huang/cosa/results/random_search/')
            dataset_path = output_dir_bo / f'seed_{seed}.csv'
            best_metric, _ = parse_best_results(dataset_path, num_sample)
            data_entry.append(best_metric)
        result_data[entry_name] = data_entry
           
    # print(result_data)
    results_path = 'results.json'
    utils.store_json(results_path, result_data, indent=2)


if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    base_arch_path = pathlib.Path(args.base_arch_path).resolve()
    arch_dir = pathlib.Path(args.arch_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    config_dir = pathlib.Path(args.config_dir).resolve()
    gen_results_dataset(base_arch_path, arch_dir, output_dir, config_dir)
    
    # parse_best_results('dse_output_dir_dataset_500/dataset_optimal_search_Newton.csv', 10)
    #parse_search_results()
    
    # mesh x [3, 4, 5, 8, 10, 12, 14] 
    # 4,256,256,1,128,256,128,16384,64,4096,1,32768
    # hw_config = [4,256,256,16384,4096,32768]
    # eval_result = eval(hw_config, base_arch_path, arch_dir, output_dir, config_prefix='', arch_v3=False, unique_sum=True, workload_dir=None)
    # print(eval_result)
    # print(eval_result[0] * eval_result[1])
