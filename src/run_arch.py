#!/usr/bin/env python3 
import argparse
import pathlib
import copy
import itertools
import os
import subprocess
import re

import utils
from cosa_input_objs import Arch

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
                        default='gen_arch',
                        )
    parser.add_argument('-bap',
                        '--base_arch_path',
                        type=str,
                        help='Hardware Architecture Path',
                        default=f'{_COSA_DIR}/configs/arch/simba_v3.yaml',
                        )
    return parser


def gen_arch_yaml(base_arch_path, output_dir):
    # Get base arch dictionary
    base_arch = utils.parse_yaml(base_arch_path)

    # Create directory for new arch files, if necessary
    new_arch_dir = output_dir
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


def gen_data(new_arch_dir, output_dir):
    # Get all arch files
    arch_files = list(new_arch_dir.glob('arch_pe*_v3.yaml'))
    arch_files.sort()

    # arch_files = arch_files[:1]
    
    processes = []
    # Start schedule generation script for each layer on each arch
    for arch_file in arch_files:

        cmd = ["python", "run_dnn_models.py", "--output_dir", str(output_dir), "--arch_path", arch_file]

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

def fetch_data(new_arch_dir, output_dir):
    # Get all arch files
    arch_files = new_arch_dir.glob('arch_pe*_v3.yaml')
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
                "energy": total_energy
            }
        
        db[arch.config_str()] = arch_db

        if len(db) % 100 == 0:
            print(f"Fetched data for {len(db)} arch")
    
    utils.store_json("all_arch.json", db)

if __name__ == "__main__":
    parser = construct_argparser()
    args = parser.parse_args()

    base_arch_path = pathlib.Path(args.base_arch_path).resolve()
    arch_dir = pathlib.Path(args.arch_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()

    # gen_arch_yaml(base_arch_path, arch_dir)
    gen_data(arch_dir, output_dir)
    # fetch_data(new_arch_dir, output_dir)
