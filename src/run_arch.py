import pathlib
import copy
import itertools
import os
import subprocess
import json

import utils
from cosa_input_objs import Arch

def gen_arch_yaml(base_arch_name):
    # Get base arch dictionary
    base_arch_path = pathlib.Path("/scratch/charleshong/matchlib/cmod/unittests/HybridRouterTopMeshTCGB/timeloop_configs/arch/").resolve() / (base_arch_name + ".yaml")
    base_arch_dict = utils.parse_yaml(base_arch_path)

    # Create directory for new arch files, if necessary
    new_arch_dir = pathlib.Path("timeloop_configs/gen_arch/")
    new_arch_dir.mkdir(parents=True, exist_ok=True)

    buf_multipliers = [0.5, 1, 2, 4]
    buf_multipliers_perms = [p for p in itertools.product(buf_multipliers, repeat=4)]

    for pe_multiplier in [0.25, 1, 4]:
        for mac_multiplier in [0.25, 1, 4]:
            for buf_multipliers_perm in buf_multipliers_perms:
                new_arch_dict = copy.deepcopy(base_arch_dict)
                
                # Get nested dictionaries
                base_arith = base_arch_dict["arch"]["arithmetic"]
                new_arith = new_arch_dict["arch"]["arithmetic"]
                base_storage = base_arch_dict["arch"]["storage"]
                new_storage = new_arch_dict["arch"]["storage"]

                arch_invalid = False
                # PE and buffer
                new_arith["meshX"] = int(base_arith["meshX"] * pe_multiplier)
                for i in range(5): # Ignoring DRAM
                    if "meshX" in new_storage[i]:
                        new_storage[i]["meshX"] = int(base_storage[i]["meshX"] * pe_multiplier)

                        # Check whether meshX divides num instances of all buffers
                        if new_storage[i]["instances"] % new_storage[i]["meshX"] != 0:
                            print("Arch invalid")
                            print("Instances:", new_storage[i]["instances"])
                            print("meshX:", new_storage[i]["meshX"])
                            arch_invalid = True

                    if i != 0: # Ignoring registers
                        new_storage[i]["entries"] = int(base_storage[i]["entries"] * buf_multipliers_perm[i-1])
                    
                if arch_invalid:
                    continue

                # MAC
                new_arith["instances"] = int(base_arith["instances"] * mac_multiplier)
                # Set registers to match MACs
                new_storage[0]["instances"] = new_arith["instances"]

                # Construct filename for new arch
                config_str = "arch" + "_pe" + str(pe_multiplier) +   \
                                      "_mac" + str(mac_multiplier) + \
                                      "_buf"
                for multiplier in buf_multipliers_perm:
                    config_str += "_" + str(multiplier)
                config_str += ".yaml"

                # Save new arch
                new_arch_path = new_arch_dir.resolve() / config_str
                utils.store_yaml(new_arch_path, new_arch_dict)

def gen_data(new_arch_dir, output_dir):
    # Get all arch files
    arch_files = list(new_arch_dir.glob('arch_pe*.yaml'))
    arch_files.sort()

    # a12: run on first 1152 sorted elements
    arch_files = arch_files[:1152]

    # a11: run on last 1152 sorted elements
    # arch_files = arch_files[1152:]

    processes = []
    # Start schedule generation script for each layer on each arch
    for arch_file in arch_files:
        print(arch_file)

        cmd = ["python", "run_dnn_models.py", "--output_dir", str(output_dir), "--arch_path", \
               arch_file, "--model", "resnet50"]

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

def fetch_data(new_arch_dir, output_dir):
    # Get all arch files
    arch_files = new_arch_dir.glob('arch_pe*.yaml')

    processes = []
    # Fetch data into DB
    for arch_file in arch_files:
        # Get each file's data
        arch_name = os.path.basename(arch_file)
        cycle_json = output_dir / arch_name / f"results_{arch_name}_cycle.json"
        cycle_dict = utils.parse_json(cycle_json)
        print(cycle_dict)
        return

if __name__ == "__main__":
    # gen_arch_yaml("simba_final_input_modified")
    new_arch_dir = pathlib.Path("/scratch/charleshong/matchlib/cmod/unittests/HybridRouterTopMeshTCGB/timeloop_configs/gen_arch/")
    output_dir = pathlib.Path("output_dir")
    gen_data(new_arch_dir, output_dir)
    # fetch_data(new_arch_dir, output_dir)
