#!/usr/bin/env python3 

from bayes_opt import BayesianOptimization
import argparse
import logging
import os
import pathlib
import math

import utils
from cosa import run_timeloop
from cosa_input_objs import Prob
from run_arch import gen_arch_yaml_from_config, gen_data, gen_dataset_col_title, append_dataset_csv, parse_results, fetch_arch_perf_data 
from eval_dse import discretize_config

#### MUST DOWNGRADE scikit-learn to 0.22 , by ``pip install scikit-learn==0.22''

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
    parser.add_argument('-a',
                        '--arch_dir',
                        type=str,
                        help='Generated Archtecture Folder',
                        default='arch_bo',
                        )
    parser.add_argument('-bap',
                        '--base_arch_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{_COSA_DIR}/configs/arch/simba_dse.yaml',
                        )
    parser.add_argument(
                        '--random_seed',
                        type=int,
                        help='Random Seed',
                        default=1,
                        )
    parser.add_argument(
                        '--num_samples',
                        type=int,
                        help='Number of total samples',
                        default=2100,
                        )

    parser.add_argument(
                        '--model',
                        type=str,
                        help='Target DNN Model',
                        default='resnet50',
                        )

    return parser


def eval(hw_config, base_arch_path, arch_dir, output_dir, dataset_path, model, config_prefix='', arch_v3=False, unique_sum=True, workload_dir='../configs/workloads'):
    hw_config = discretize_config(hw_config)
    config_yaml_str = gen_arch_yaml_from_config(base_arch_path, arch_dir, hw_config, config_prefix, arch_v3=arch_v3)
    glob_str = config_yaml_str
    config_str = config_yaml_str.replace('.yaml', '')
    gen_data(arch_dir, output_dir, glob_str, model=model)
    cycle, energy, area = parse_results(output_dir, config_str, unique_sum, model=model)
    data = fetch_arch_perf_data(arch_dir, output_dir, glob_str, arch_v3, mem_levels=5, model=model)
    append_dataset_csv(data, dataset_path)
    return (cycle, energy, area)


def bo(base_arch_path, arch_dir, output_dir, num_samples, model='resnet50', init_samples=0, random_seed=1):
    assert(num_samples > init_samples)

    dataset_path = output_dir / f'dataset_{model}_s{random_seed}.csv'
    with open(dataset_path,  'w') as f:
        key = gen_dataset_col_title()
        f.write(f'{key}\n')

    pbounds = {}
    bounds = [64, 32, 256, 256, 4096, 256]
    # scales = [1, 128, 1, 2**8, 1, 2**10]
    scales = [1, 1, 1, 2**8, 1, 2**10]
    
    for i, bound in enumerate(bounds):
        pbounds[i] = (1, bound)

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,
        random_state=random_seed,
        #random_state=888,
        #random_state=103495333,
        #random_state=56781234,
        #random_state=1234,
    )
    
    from bayes_opt import UtilityFunction
    utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)

    target_scale = 1e+14

    init_data = []
    for iteration in range(init_samples):
        next_point_to_probe = optimizer.suggest(utility)
        print("Next point to probe is:", next_point_to_probe)
        hw_config = []
        for i in range(len(bounds)):
            hw_config.append(next_point_to_probe[i] * scales[i])

        cycle, energy, area = eval(hw_config, base_arch_path, arch_dir, output_dir, dataset_path, model)
        target = cycle * energy / target_scale 
        init_data.append((next_point_to_probe, target))

    for iteration in range(init_samples):
        next_point_to_probe, target = init_data[iteration]
        optimizer.register(
            params=next_point_to_probe,
            target=target,
        )

    for iteration in range(num_samples-init_samples):

        next_point_to_probe = optimizer.suggest(utility)
        print("Next point to probe is:", next_point_to_probe)
        
        hw_config = []
        for i in range(len(bounds)):
            hw_config.append(next_point_to_probe[i] * scales[i])

        cycle, energy, area = eval(hw_config, base_arch_path, arch_dir, output_dir, dataset_path, model)

        target = cycle * energy / target_scale
        
        print("Next point to probe is:", next_point_to_probe)
        print("Target:", target)
        optimizer.register(
            params=next_point_to_probe,
            target=target,
        )
        print(f'it: {iteration}, {target}, {next_point_to_probe}')
        
        
if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    base_arch_path = pathlib.Path(args.base_arch_path).resolve()

    model = args.model
    random_seed = args.random_seed

    output_dir = args.output_dir
    output_dir = f'{output_dir}_{model}_s{random_seed}' 
    output_dir = pathlib.Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arch_dir = args.arch_dir
    arch_dir = f'{arch_dir}_{model}_s{random_seed}' 
    arch_dir = pathlib.Path(arch_dir).resolve()

    # mesh x [3, 4, 5, 8, 10, 12, 14] 
    # 4,256,256,1,128,256,128,16384,64,4096,1,32768
    #hw_config = [4,256,256,256,16384,4096,32768]
    #eval_result = eval(hw_config, arch_dir, output_dir, search_algo='', opt_algo='', arch_v3=False, unique_sum=True, workload_dir=None)
    #print(eval_result)
    # print(eval_result[0] * eval_result[1])
    num_samples = args.num_samples
    bo(base_arch_path, arch_dir, output_dir, num_samples, random_seed=random_seed, model=model)
