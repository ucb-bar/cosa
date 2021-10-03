#!/usr/bin/env python3 
import argparse
import logging
import os
import pathlib
import math

import utils

from eval_dse import parse_best_results

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
    seeds = [1234]
    nzs = [2, 4]
    for nz in nzs:
        for seed in seeds: 
            for search_algo in search_algos: 
                for opt_algo in opt_algos: 
                    entry_name = f'vae_{search_algo}_{opt_algo}_{seed}_{nz}'
                    if seed == 1:
                        seed_str=''
                    else:
                        seed_str=f'_seed_{seed}'
                    data_entry = []
                    if nz != 2: 
                        nz_str = f'_nz_{nz}'
                    else:
                        nz_str = ''
                    for num_sample in samples: 
                        output_dir = path / f'dse_output_dir_dataset_{num_sample}{seed_str}{nz_str}'  
                        dataset_path = output_dir / f'dataset_{search_algo}_{opt_algo}.csv'
                        best_metric, _ = parse_best_results(dataset_path, num_sample)
                        data_entry.append(best_metric)
                    result_data[entry_name] = data_entry

    # parse BO results 
#     seeds = [1, 1234]
#     for seed in seeds:
#         entry_name = f'bo_{seed}'
#         data_entry = []
#         for num_sample in samples:
#             output_dir_bo = pathlib.Path('/nscratch/qijing.huang/cosa/results/bo_search/')
#             dataset_path = output_dir_bo / f'dataset_s{seed}.csv'
#             best_metric, _ = parse_best_results(dataset_path, num_sample)
#             data_entry.append(best_metric)
#         result_data[entry_name] = data_entry
#             
#     seeds = [6, 7]
#     for seed in seeds:
#         entry_name = f'random_{seed}'
#         data_entry = []
#         for num_sample in samples:
#             output_dir_bo = pathlib.Path('/nscratch/qijing.huang/cosa/results/random_search/')
#             dataset_path = output_dir_bo / f'seed_{seed}.csv'
#             best_metric, _ = parse_best_results(dataset_path, num_sample)
#             data_entry.append(best_metric)
#         result_data[entry_name] = data_entry
           
    # print(result_data)
    #results_path = 'results_vae.json'
    results_path = 'results_vae_new.json'
    utils.store_json(results_path, result_data, indent=2)


if __name__ == "__main__":
    parse_search_results()
