#!/usr/bin/env python3 
import argparse
import pathlib
import copy
import itertools
import os
import subprocess
import re

import utils
from cosa_input_objs import Arch, Prob

_COSA_DIR = os.environ['COSA_DIR']


def construct_argparser():

    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-o',
                        '--output_dir',
                        type=str,
                        help='Output Folder',
                        default='output_dir_dataset',
                        )
    parser.add_argument('-a',
                        '--arch_dir',
                        type=str,
                        help='Generated Archtecture Folder',
                        default='gen_arch_dataset',
                        )
    parser.add_argument('-bap',
                        '--base_arch_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{_COSA_DIR}/configs/arch/simba_dse_v3.yaml',
                        )
    return parser


def gen_arch_yaml_from_config(base_arch_path, arch_dir, hw_configs, config_prefix, arch_v3=False):
    # Get base arch dictionary
    base_arch = utils.parse_yaml(base_arch_path)

    # Create directory for new arch files, if necessary
    new_arch_dir = arch_dir
    new_arch_dir.mkdir(parents=True, exist_ok=True)
    new_arch = copy.deepcopy(base_arch)

    # parse hw config
    arith_meshX,arith_ins,mem1_ent,mem2_ent,mem3_ent,mem4_ent = hw_configs 
 
    if arch_v3: 
        base_arch_dict = base_arch["architecture"]["subtree"][0]["subtree"][0]
        new_arch_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]
        raise 
    else:
	# Get nested dictionaries
        base_arith = base_arch["arch"]["arithmetic"]
        new_arith = new_arch["arch"]["arithmetic"]
        base_storage = base_arch["arch"]["storage"]
        new_storage = new_arch["arch"]["storage"]
        new_arith["meshX"] = arith_meshX
        new_arith["instances"] = int(arith_ins) * 128
        new_storage[0]["instances"] = int(arith_ins) * 128 
        new_storage[1]["entries"] = mem1_ent
        new_storage[2]["entries"] = mem2_ent
        new_storage[3]["entries"] = mem3_ent
        new_storage[4]["entries"] = mem4_ent

        for i in range(5):
            if "meshX" in new_storage[i]:
                new_storage[i]["meshX"] = arith_meshX
            
    hw_configs_arr = [str(i) for i in hw_configs]
    hw_configs_str = "_".join(hw_configs_arr)

    # Construct filename for new arch
    config_str = get_hw_config_str(hw_configs, config_prefix)

    # Save new arch
    new_arch_path = new_arch_dir.resolve() / config_str
    utils.store_yaml(new_arch_path, new_arch)
    return config_str


def get_hw_config_str(hw_configs, config_prefix, arch_v3=False):
    hw_configs_arr = [str(i) for i in hw_configs]
    hw_configs_str = "_".join(hw_configs_arr)

    # Construct filename for new arch
    config_str = f"arch_{config_prefix}"
    config_str += hw_configs_str 
    if arch_v3: 
        config_str += "_v3"
    config_str += ".yaml"
    return config_str


def gen_arch_yaml(base_arch_path, arch_dir):
    # Get base arch dictionary
    base_arch = utils.parse_yaml(base_arch_path)

    # Create directory for new arch files, if necessary
    new_arch_dir = arch_dir
    new_arch_dir.mkdir(parents=True, exist_ok=True)

    buf_multipliers = [0.5, 1, 2, 4]
    buf_multipliers_perms = [p for p in itertools.product(buf_multipliers, repeat=4)]

    for pe_multiplier in [0.25, 1, 4]:
        for mac_multiplier in [0.25, 1, 4]:
            for buf_multipliers_perm in buf_multipliers_perms:
                new_arch = copy.deepcopy(base_arch)
                base_arch_dict = base_arch["architecture"]["subtree"][0]["subtree"][0]
                new_arch_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]
                
                print(f"{pe_multiplier} {mac_multiplier} {buf_multipliers_perm}")
                base_meshX_str =  base_arch_dict["subtree"][0]["name"]
                m = re.search("PE\[0..(\S+)\]", base_meshX_str)
                if not m:
                    raise ValueError("Wrong mesh-X specification.")
                base_meshX = int(m.group(1)) + 1
                new_meshX = int(base_meshX * pe_multiplier) - 1 
                new_arch_dict["subtree"][0]["name"] = f"PE[0..{new_meshX}]" 

                # Get nested dictionaries
                base_arith = base_arch_dict["subtree"][0]["local"][4]["attributes"] 
                new_arith = new_arch_dict["subtree"][0]["local"][4]["attributes"]


                base_storage = base_arch_dict["subtree"][0]["local"]
                new_storage = new_arch_dict["subtree"][0]["local"]

                arch_invalid = False

                # PE and buffer
                new_arith["meshX"] = int(base_arith["meshX"] * pe_multiplier)
                
                # PE buffers 
                for i in range(3): # Ignoring DRAM
                    if "meshX" in new_storage[i]["attributes"]:
                        new_storage[i]["attributes"]["meshX"] = int(base_storage[i]["attributes"]["meshX"] * pe_multiplier)

                        # Check whether meshX divides num instances of all buffers
                        if new_storage[i]["attributes"]["instances"] % new_storage[i]["attributes"]["meshX"] != 0:
                            print("Arch invalid")
                            print("Instances:", new_storage[i]["attributes"]["instances"])
                            print("meshX:", new_storage[i]["attributes"]["meshX"])
                            arch_invalid = True

                    # if i != 0: # Ignoring registers
                    new_storage[i]["attributes"]["entries"] = int(base_storage[i]["attributes"]["entries"] * buf_multipliers_perm[i])

                # global buffer
                base_gb_dict = base_arch_dict["local"][0]
                new_gb_dict = new_arch_dict["local"][0]
                new_gb_dict["attributes"]["entries"] = int(base_gb_dict["attributes"]["entries"] * buf_multipliers_perm[3])
                    
                if arch_invalid:
                    continue

                # MAC
                new_arith["instances"] = int(base_arith["instances"] * mac_multiplier)

                # Set registers to match MACs
                new_storage[3]["attributes"]["instances"] = new_arith["instances"]
                new_storage[3]["attributes"]["meshX"] = int(base_storage[3]["attributes"]["meshX"] * pe_multiplier)

                # Construct filename for new arch
                config_str = "arch" + "_pe" + str(pe_multiplier) +   \
                                      "_mac" + str(mac_multiplier) + \
                                      "_buf"
                for multiplier in buf_multipliers_perm:
                    config_str += "_" + str(multiplier)
                config_str += "_v3.yaml"
                
                # Save new arch
                new_arch_path = new_arch_dir.resolve() / config_str
                utils.store_yaml(new_arch_path, new_arch)


def gen_dataset_col_title():
    col_str = ['name', 'unique_cycle_sum', 'unique_energy_sum', 'arith_meshX', 'arith_ins']
    for i in range(5):
        # col_str.extend([f'mem{i}_meshX', f'mem{i}_ins', f'mem{i}_ent'])
        col_str.extend([f'mem{i}_ins', f'mem{i}_ent'])
    key = ','.join(col_str)
    return key


def gen_dataset_csv(data, dataset_path):
    with open(dataset_path,  'w') as f:
        key = gen_dataset_col_title()
        f.write(f'{key}\n')
        for d in data:
            key = d[0]
            col_str = ','.join(d[1])
            f.write(f'{key},{col_str}\n')


def append_dataset_csv(data, dataset_path):
    with open(dataset_path,  'a') as f:
        for d in data:
            key = d[0]
            col_str = ','.join(d[1])
            f.write(f'{key},{col_str}\n')
            

def parse_results(output_dir, config_str, unique_sum=True, model='resnet50', layer_idx=None, workload_dir='../configs/workloads'):
    # if network is None, return sum of 4 networks
    # if layer is None, reuturn sum of specific network
    cycle_path = output_dir / f'results_{config_str}_cycle.json'
    energy_path = output_dir / f'results_{config_str}_energy.json'
    area_path = output_dir /f'results_{config_str}_area.json'
    print(f'path: {cycle_path}') 
    if layer_idx: 
        workload_dir = pathlib.Path(workload_dir).resolve()
        model_dir = workload_dir / (model+'_graph')
        layer_def_path = model_dir / 'unique_layers.yaml'
        layers = utils.parse_yaml(layer_def_path)
        layer = list(layers)[layer_idx]
        prob_path = model_dir / (layer + '.yaml')
        prob = Prob(prob_path)
        prob_key = prob.config_str()

        cycle = utils.parse_json(cycle_path)[model][prob_key]
        energy = utils.parse_json(energy_path)[model][prob_key]
        area = utils.parse_json(area_path)[model][prob_key]
    else: 
        if unique_sum: 
            try:
                cycle = sum(utils.parse_json(cycle_path)[model].values())
                energy = sum(utils.parse_json(energy_path)[model].values())
                area = list(utils.parse_json(area_path)[model].values())[0]
            except:
                raise
        else:
            # Load aggregated results JSON files
            cycle_dict = utils.parse_json(cycle_path)
            energy_dict = utils.parse_json(energy_path)
            area_dict = utils.parse_json(area_path)

            # Load the layer count file for the selected model
            workload_dir = pathlib.Path(workload_dir).resolve()
            model_dir = workload_dir / (model+'_graph')
            layer_count_path = model_dir / ('layer_count.yaml')
            layer_counts_model = utils.parse_yaml(layer_count_path)
            
            # Compute total cycle count/energy
            cycle = total_layer_values(cycle_dict[model], layer_counts_model)
            energy = total_layer_values(energy_dict[model], layer_counts_model)
            
            # Just one value for area
            area = list(utils.parse_json(area_path)['resnet50'].values())[0]
    return cycle, energy, area


def fetch_arch_perf_data(new_arch_dir, output_dir, glob_str='arch_pe*_v3.yaml', arch_v3=False, mem_levels=5, model_cycles=False):
    # Get all arch files
    arch_files = list(new_arch_dir.glob(glob_str))
    # arch_files.sort()
    data = []
    
    min_cycle_energy = None
    for arch_file in arch_files: 
        base_arch_str = arch_file.name 
        m = re.search("(\S+).yaml", base_arch_str)
        if not m:
            raise ValueError("Wrong config string format.")
        config_str = m.group(1)

        new_arch = utils.parse_yaml(arch_file)
        config_v3_str = ""
        if arch_v3: 
            base_arch_dict = new_arch["architecture"]["subtree"][0]["subtree"][0]
            base_meshX_str =  base_arch_dict["subtree"][0]["name"]
            m = re.search("PE\[0..(\S+)\]", base_meshX_str)
            if not m:
                raise ValueError("Wrong mesh-X specification.")
            base_meshX = int(m.group(1)) + 1

            base_arith = base_arch_dict["subtree"][0]["local"][4]["attributes"] 
            base_storage = base_arch_dict["subtree"][0]["local"]
            data_entry = [str(base_meshX), str(base_arith["instances"])]
            
            for i in reversed(range(4)): 
                data_entry.extend([str(base_storage[i]["attributes"]["instances"]),str(base_storage[i]["attributes"]["entries"])])
            base_gb_dict = base_arch_dict["local"][0]
            data_entry.extend([str(base_gb_dict["attributes"]["instances"]), str(base_gb_dict["attributes"]["entries"])])
        else:
            # Get nested dictionaries
            new_arith = new_arch["arch"]["arithmetic"]
            new_storage = new_arch["arch"]["storage"]

            data_entry = [str(new_arith["meshX"]), str(new_arith["instances"]), ]
            for i in range(mem_levels): # Ignoring DRAM
                data_entry.extend([str(new_storage[i]["instances"]), str(new_storage[i]["entries"])])

        # Get the labels 
        cycle, energy, area = parse_results(output_dir, config_str, unique_sum=True, workload_dir='../configs/workloads')
        edp = cycle * energy
        adp = area * cycle
        if cycle * energy > 0: 
            if min_cycle_energy: 
                if edp < min_cycle_energy:
                    print(config_str)
                    min_cycle_energy = edp
            else:
                min_cycle_energy = edp
            print(f'min_cycle_energy: {min_cycle_energy}')

            # data_entry = [str(cycle), str(energy)] + [str(area), str(edp), str(adp)]  + data_entry
            data_entry = [str(cycle), str(energy)] + data_entry
            data.append((config_str, data_entry))
    return data


def gen_dataset(new_arch_dir, output_dir, glob_str='arch_pe*_v3.yaml', arch_v3=False, mem_levels=5, model_cycles=False, postfix=''):
    config_str = glob_str.replace('_*.yaml', '')
    dataset_path = output_dir / f'dataset{postfix}.csv'
    print(dataset_path)

    data = fetch_arch_perf_data(new_arch_dir, output_dir, glob_str=glob_str, arch_v3=arch_v3, mem_levels=mem_levels, model_cycles=model_cycles)
    # print(data)
    gen_dataset_csv(data, dataset_path)
    return min_cycle_energy


def gen_dataset_per_layer(new_arch_dir, output_dir, glob_str='arch_pe*_v3.yaml', arch_v3=False, mem_levels=5, model_cycles=False, postfix=''):
    config_str = glob_str.replace('_*.yaml', '')
    dataset_path = output_dir / f'dataset{postfix}.csv'
    print(dataset_path)

    data = fetch_arch_perf_data(new_arch_dir, output_dir, glob_str=glob_str, arch_v3=arch_v3, mem_levels=mem_levels, model_cycles=model_cycles)
    # print(data)
    gen_dataset_csv(data, dataset_path)
    return min_cycle_energy


def gen_data(new_arch_dir, output_dir, glob_str='arch_pe*_v3.yaml'):
    # Get all arch files
    arch_files = list(new_arch_dir.glob(glob_str))
    arch_files.sort()
    print(arch_files)

    # arch_files = arch_files[:1]
    
    processes = []
    # Start schedule generation script for each layer on each arch
    for arch_file in arch_files:

        cmd = ["python", "run_dnn_models.py", "--output_dir", str(output_dir), "--arch_path", arch_file] #"--model", "resnet50"]

        process = subprocess.Popen(cmd)
        processes.append(process)

        # Limit number of active processes
        while len(processes) >= 10:
            # Iterate backward so the list of processes doesn't get messed up
            for i in range(len(processes)-1, -1, -1):
                if processes[i].poll() is not None: # If process has terminated
                    processes.pop(i)
    
    # Wait for schedule generation to finish
    for process in processes:
        process.wait()


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


def fetch_data(new_arch_dir, output_dir, glob_str='arch_pe*_v3.yaml'):
    # Get all arch files
    arch_files = new_arch_dir.glob(glob_str)
    workload_dir = pathlib.Path('../configs/workloads').resolve()

    db = {}
    layer_counts = {}
    # Fetch data into DB
    for arch_path in arch_files:
        # Get each file's data
        arch_name = os.path.basename(arch_path).split(".yaml")[0]
        cycle_json = output_dir / f"results_{arch_name}_cycle.json"
        energy_json = output_dir / f"results_{arch_name}_energy.json"
        try:
            cycle_dict = utils.parse_json(cycle_json)
            energy_dict = utils.parse_json(energy_json)
        except:
            # Data missing for some reason
            continue
        
        arch = Arch(arch_path)
        arch_db = {}
        
        for model in cycle_dict:
            if model not in layer_counts:
                model_dir = workload_dir / (model+'_graph')
                layer_count_path = model_dir / ('layer_count.yaml')
                layer_counts[model] = utils.parse_yaml(layer_count_path)
            
            total_cycle = total_layer_values(cycle_dict[model], layer_counts[model])
            total_energy = total_layer_values(energy_dict[model], layer_counts[model])
            arch_db[model] = {
                "cycle": total_cycle,
                "energy": total_energy,
                "cycle_energy_prod": total_energy * total_cycle,
            }
        
        db[arch.config_str()] = arch_db

        if len(db) % 100 == 0:
            print(f"Fetched data for {len(db)} arch")
    
    utils.store_json(output_dir / "all_arch.json", db)
    

if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    base_arch_path = pathlib.Path(args.base_arch_path).resolve()
    arch_dir = pathlib.Path(args.arch_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    # gen_arch_yaml(base_arch_path, arch_dir)
    # gen_data(arch_dir, output_dir)
    gen_dataset(arch_dir, output_dir, arch_v3=True, postfix='_v3')
    # fetch_data(new_arch_dir, output_dir)

