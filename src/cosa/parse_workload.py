import collections
import copy
import logging
import os
import pathlib
import re
import xml.etree.ElementTree as ET

import numpy as np
import cosa.utils
from cosa.utils import OrderedDefaultdict

logger = logging.getLogger(__name__)
logger.setLevel(logging.NOTSET)  # capture everything
# logger.setLevel(logging.) # capture everything
logger.disabled = True

all_var_names = ["Weights", "Inputs", "Outputs"]

var_dep = {
    "Weights": [],
    "Inputs": [],
    "Outputs": [],  # Partial sum does not dependent on any var
    "Outputs_Update": ["Weights", "Inputs", "Outputs"]
# Must wait until last output is written to the buf? maybe delete Outputs
}

var_bits = {
    "Weights": 8,
    "Inputs": 8,
    "Outputs": 24,
    "Outputs_Store": 24
}

dim_var = {
    0: ['Weights'],  # R
    1: ['Weights'],  # S
    2: ['Outputs', 'Inputs'],  # P
    3: ['Outputs', 'Inputs'],  # Q
    4: ['Weights', 'Inputs'],  # C
    5: ['Weights', 'Outputs'],  # K
    6: ['Inputs', 'Outputs'],  # N
}


def xml2dict(t):
    d = {t.tag: {} if t.attrib else None}
    children = list(t)
    if children:
        dd = OrderedDefaultdict(list)
        for dc in map(xml2dict, children):
            for k, v in dc.items():
                dd[k].append(v)
        d = {t.tag: {k: v[0] if len(v) == 1 else v
                     for k, v in dd.items()}}
    if t.attrib:
        d[t.tag].update(('@' + k, v)
                        for k, v in t.attrib.items())
    if t.text:
        text = t.text.strip()
        if children or t.attrib:
            if text:
                d[t.tag]['#text'] = text
        else:
            d[t.tag] = text
    return d


def addr_to_str(addr):
    addr_arr = [str(a) for a in addr]
    return "_".join(addr_arr)


# Construct addrs dict from timeloop info
# Create a dictionary for different addresses, i would be the PE idx 
def construct_addrs_dict(buf):
    addrs = collections.OrderedDict()

    for i, addr in enumerate(buf):
        addr_key = addr_to_str(addr)
        if addr_key in addrs.keys():
            addrs[addr_key].append(i)
        else:
            addrs[addr_key] = [i]
    return addrs


def get_summary_info(stats_file):
    summary = {}
    with open(stats_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        # m = re.match(r"Energy: (.*) uJ", line)
        m = re.match(r"Total topology energy: (.*) pJ", line)
        if m:
            energy = m.group(1)
            summary['energy'] = float(energy)
        else:
            m = re.match(r"Max topology cycles: (.*)", line)
            if m:
                cycle = m.group(1)
                summary['cycle'] = int(cycle)
    return summary


"""

Returns: 
subnest_info = {
    subnest = {
        'Registers': [{'end': '1', 'spacetime_dimension': '0', '@class_id': '11', '@tracking_level': '0', 'start': '0', 'stride': '1', '@version': '0', 'dimension': '6'}, ...],  # arrays of subnest info 
        
    },
    bufsize = {
        'Registers': ('Registers', [0, 1, 0]), # buffered size for Weights, Inputs, Outputs
        ...
    }

}

"""


def get_subnest_info(xml_file):
    logger.info("================ Parse Subnest Info from {0} ================".format(xml_file))
    xml_path = pathlib.Path(xml_file)
    stats_file = xml_path.parent / xml_path.name.replace('map+stats.xml', 'stats.txt')
    summary_info = get_summary_info(stats_file)

    subnest_info = {}
    subnest_info['energy'] = summary_info['energy']
    subnest_info['cycle'] = summary_info['cycle']
    tree = ET.parse(xml_file);
    root = tree.getroot()

    timeloop_dict = xml2dict(root)
    # print(timeloop_dict)

    # print(timeloop_dict['boost_serialization']['best_mapped_engine']['topology_']['levels_']['item'])
    arch = timeloop_dict['boost_serialization']['engine']['topology_']['levels_']['item']
    arith = arch[0]
    # print(arith['px']['@class_name'])
    subnest_info['pe_cycle'] = int(arith['px']['cycles_'])
    subnest_info['pe_energy'] = float(arith['px']['energy_'])

    bufsize = collections.OrderedDict()
    subnest = collections.OrderedDict()

    for buf in arch[1:]:
        level_name = buf['px']['specs_']['LevelSpecs']['level_name']
        bufsize_arr = buf['px']['stats_']['utilized_capacity']['PerDataSpace']['item']
        bufsize_arr = [int(size) for size in bufsize_arr]

        bufsize[level_name] = bufsize_arr  # weight, input, output
        subnest_attr = buf['px']['subnest_']['item']
        if not isinstance(subnest_attr, list):
            subnest_attr = [subnest_attr]

        subnest[level_name] = subnest_attr

    subnest_info['bufsize'] = bufsize
    subnest_info['subnest'] = subnest
    return subnest_info


def get_num_spatial_cores(subnest, start_level, end_level):
    num_spatial_cores = 1
    prev_spatial = True

    # Get outer loop traffic iters from WeightInputBuffer 
    for subnest in list(subnest.values())[start_level:end_level]:
        # Convert everything to a list
        for loop in subnest:
            logger.info("Loop Levels (inner to outer): {0}".format(loop))
            loop_dimension = int(loop['dimension'])
            start = int(loop['start'])
            end = int(loop['end'])
            stride = int(loop['stride'])
            spacetime_dimension = int(loop['spacetime_dimension'])
            size = (end - start)
            if spacetime_dimension != 0:  # If meshX is not 0
                logger.info("\tSpatial Partiton with size {0}".format(size))
                num_spatial_cores *= size
                assert (prev_spatial)  # Make sure spatial loops are the innermost loops
            else:
                prev_spatial = False

    return num_spatial_cores


def get_pe_cycle(subnest, start_level):
    pe_cycle = 1
    # Get inner loop traffic size 
    for subnest in list(subnest.values())[:start_level]:
        for loop in subnest:
            loop_dimension = int(loop['dimension'])
            start = int(loop['start'])
            end = int(loop['end'])
            stride = int(loop['stride'])
            spacetime_dimension = int(loop['spacetime_dimension'])
            size = (end - start)

            # Get the for loop cycles
            if spacetime_dimension == 0:
                pe_cycle *= size
    return pe_cycle


def init_data_size(subnest_info, start_level):
    data_size = {}
    subnest = subnest_info['subnest']
    bufsize = subnest_info['bufsize']

    is_outside_buf = {}
    idx_var_dict = {0: "Weights", 1: "Inputs", 2: "Outputs"}
    for var_name in all_var_names:
        data_size[var_name] = 0
        is_outside_buf[var_name] = False

    # Get inner loop traffic size 
    for i, subnest in enumerate(list(subnest.values())[:start_level]):

        # if the loop is outside the buffer, use the loop bound
        for var_name in idx_var_dict.values():
            if is_outside_buf[var_name]:
                for loop in subnest:
                    loop_dimension = int(loop['dimension'])
                    start = int(loop['start'])
                    end = int(loop['end'])
                    stride = int(loop['stride'])
                    spacetime_dimension = int(loop['spacetime_dimension'])
                    size = (end - start)

                    # Lookup the var affected by the current dim
                    var_arr = dim_var[loop_dimension]
                    if var_name in var_arr:
                        data_size[var_name] *= size
        # else use the timeloop figure
        for var_idx, mem_size in enumerate(list(bufsize.values())[i]):
            if mem_size != 0:
                var_name = idx_var_dict[var_idx]
                data_size[var_name] = mem_size
                is_outside_buf[var_name] = True
    for var_name in idx_var_dict.values():
        logger.info("Data size:\n\t{}={}".format(var_name, data_size[var_name]))
        # print("Data size:\n\t{}={}".format(var_name, schedule['data_size'][var_name]))
    return data_size


def get_iter_space(subnest, start_level, end_level):
    loops = []
    loop_iter = []
    for subnest in list(subnest.values())[start_level:end_level]:
        for loop in subnest:
            item = {}
            item['dimension'] = int(loop['dimension'])
            item['start'] = int(loop['start'])
            item['end'] = int(loop['end'])
            item['stride'] = int(loop['stride'])
            item['spacetime_dimension'] = int(loop['spacetime_dimension'])
            loops.append(item)
            loop_iter.append(item['start'])
    return (loops, loop_iter)


def is_done(loops, loop_iter):
    top_dim = len(loops) - 1
    if loop_iter[top_dim] < loops[top_dim]['end']:
        return False
    else:
        return True


def print_loop_indices(loop_iter):
    loop_indices_str = ', '.join(str(i) for i in loop_iter)
    logger.info("\tLoop Indices: {}".format(loop_indices_str))


def increment(loops, loop_iter):
    loop_carry = True
    for i, loop in enumerate(loops):
        if loop_carry:
            new_iter = loop_iter[i] + loop['stride']
            if new_iter < loop['end']:
                loop_carry = False
                loop_iter[i] = new_iter
                break
            else:
                if i < len(loops) - 1:
                    loop_iter[i] = loop['start']
                else:
                    loop_iter[i] = new_iter


def reset_var_dict_list(var_dict):
    for var in all_var_names:
        var_dict[var] = list()


def reset_var_dict_val(var_dict, val):
    for var in all_var_names:
        var_dict[var] = val


def get_temporal_dim_start(loops):
    # process spatial dims to get send data patterns
    temporal_dim_start = 0
    # loop from inner spatial to outer temporal
    for i, loop in enumerate(loops):
        # if this is spatial
        if loop['spacetime_dimension'] == 0:
            temporal_dim_start = i
            break
    return temporal_dim_start


def get_outer_temp_loopcount(subnest_info, start_level, end_level=None):
    subnest = subnest_info['subnest']
    schedule = {}

    if end_level is None:
        end_level = len(subnest_info['bufsize'])

    # get iteration space starting from PART_LEVEL 
    # loops inside of PART_LEVEL are considered 
    loops, loop_iter = get_iter_space(subnest, start_level, end_level)

    loopcount = 1
    for item in loops:
        # only count time dimension
        if item['spacetime_dimension'] == 0:
            size = (item['end'] - item['start'] + item['stride'] - 1) // item['stride']
            loopcount *= size
    return loopcount


def gen_schedule(subnest_info, start_level, end_level=None):
    logger.info("================ Generate Schedule ================")
    subnest = subnest_info['subnest']
    schedule = {}

    if end_level is None:
        end_level = len(subnest_info['bufsize'])

    schedule['num_spatial_cores'] = get_num_spatial_cores(subnest, start_level, end_level)
    schedule['pe_cycle'] = get_pe_cycle(subnest, start_level)
    data_size = init_data_size(subnest_info, start_level)

    # get iteration space starting from PART_LEVEL 
    # loops inside of PART_LEVEL are considered 
    loops, loop_iter = get_iter_space(subnest, start_level, end_level)

    temporal_dim_start = get_temporal_dim_start(loops)

    spatial_loops = loops[:temporal_dim_start]
    spatial_loop_iter = loop_iter[:temporal_dim_start]
    if not spatial_loops:
        spatial_loops = [{'dimension': start_level, 'start': 0, 'end': 1, 'stride': 1, 'spacetime_dimension': 1}]
        spatial_loop_iter = [0]
    logger.info("Spatial Loops: \n\t{}".format(spatial_loops))
    logger.info("Init Spatial Loops Iter: \n\t{}".format(spatial_loop_iter))

    temporal_loops = loops[temporal_dim_start:]
    temporal_loop_iter = loop_iter[temporal_dim_start:]
    logger.info("Temporal Loops: \n\t{}".format(temporal_loops))
    logger.info("Init Temporal Loops Iter: \n\t{}".format(temporal_loop_iter))
    schedule['temporal_loops'] = temporal_loops
    schedule['temporal_loop_iter'] = copy.deepcopy(temporal_loop_iter)

    # process spatial dims to get send data patterns
    # get the config of the buf data
    # 'Weights' : [True, False] # length = # of spatial level, item = whether the addr varies with the var
    buf_config = {}
    reset_var_dict_list(buf_config)
    for i, loop in enumerate(spatial_loops):
        dimension = loop['dimension']
        for var in all_var_names:
            buf_config[var].append(0)
            if var in dim_var[dimension]:
                buf_config[var][-1] = 1
                # For Inputs, Weights, True indicates multicast, False indicates broadcast
                # For Outputs, True indicates multicast, False indicates reduction
    logger.info("""Spatial Buffer Config (For Inputs and Weights, 1 indicates multicast, 0 indicates broadcast.
For Outputs, True indicates multicast, False indicates reduction.):\n\t{}""".format(buf_config))

    # buf_spatial stores the indices of the spatial buf data
    # 'Weights' : [(0,0), ..., ] # length = # of spatial cores, # item = addr
    buf_spatial = {}
    reset_var_dict_list(buf_spatial)
    while (not is_done(spatial_loops, spatial_loop_iter)):
        # print_loop_indices(spatial_loop_iter)
        for var in all_var_names:
            buf_spatial[var].append(copy.deepcopy(spatial_loop_iter))

        # zero out the irrelevant dims
        for i, loop in enumerate(spatial_loops):
            dimension = loop['dimension']
            for var in all_var_names:
                if not (var in dim_var[dimension]):
                    buf_spatial[var][-1][i] = 0

        increment(spatial_loops, spatial_loop_iter)
    logger.info("Buffer Spatial (Stores the Offsets for the Spatial Buffer Data):\n\t{}".format(buf_spatial))

    # 'Weights' : [True, False] # length = # of spatial level, item = whether the addr varies with the var
    iter_config = {}
    reset_var_dict_list(iter_config)
    for i, loop in enumerate(temporal_loops):
        dimension = loop['dimension']
        start = int(loop['start'])
        end = int(loop['end'])
        stride = int(loop['stride'])
        spacetime_dimension = int(loop['spacetime_dimension'])
        size = (end - start) // stride

        if size > 1:
            for var in all_var_names:
                iter_config[var].append(0)

                if var in dim_var[dimension]:
                    iter_config[var][-1] = 1
                if len(iter_config[var]) > 1:
                    if iter_config[var][-2]:
                        iter_config[var][-1] = 1

    iter_start_dim = {}
    # init all value to # of loop levels
    reset_var_dict_val(iter_start_dim, len(temporal_loops))

    for i, loop in enumerate(temporal_loops):
        for var in all_var_names:
            if iter_start_dim[var] > i:
                dimension = loop['dimension']
                if var in dim_var[dimension]:
                    iter_start_dim[var] = i
    logger.info(
        "Iter Start Dimension (Loop dimension the variable starts vary, inner to outer):\n\t{}".format(iter_start_dim))

    # find the innermost R/S/C loop that is outside output dependent prob K/N/P/Q
    output_start_dim = iter_start_dim['Outputs']
    partial_sum_dep_prob = [0, 1, 4]
    partial_sum_dep_prob_idx = len(temporal_loops)
    for i in range(output_start_dim, len(temporal_loops)):
        loop = temporal_loops[i]
        dimension = loop['dimension']
        if dimension in partial_sum_dep_prob:
            partial_sum_dep_prob_idx = i
            break
    logger.info("Outputs Start Dimension {}, Partial Sum Depend Dimension:{}".format(output_start_dim,
                                                                                     partial_sum_dep_prob_idx))

    #
    steps, iters = new_generate_temp(temporal_loops, iter_start_dim, partial_sum_dep_prob_idx)
    num_steps = len(steps['Inputs'])

    #    for key in steps.keys():
    #        #print(key)
    #        if steps[key] != dup_steps[key]:
    #            print(steps[key])
    #            print(dup_steps[key])

    cost = get_cost(iters, data_size, buf_spatial)

    logger.info("Total Number of Steps: {}".format(num_steps))

    schedule['cost'] = cost
    schedule['data_size'] = data_size
    schedule['buf_config'] = buf_config
    schedule['buf_spatial'] = buf_spatial
    schedule['iter_config'] = iter_config
    schedule['iter_start_dim'] = iter_start_dim
    schedule['steps'] = steps
    schedule['iters'] = iters
    schedule['num_steps'] = num_steps

    logger.info("Total Number of Spatial Cores: {}".format(schedule['num_spatial_cores']))
    logger.info("Total Number of PE Cycles: {}".format(schedule['pe_cycle']))

    return schedule


def get_cost(iters, data_size, buf_spatial):
    logger.info("\tGenerate_Cost")
    cost = {}
    total_cost = 0
    for var in ['Weights', 'Inputs', 'Outputs']:
        addr = construct_addrs_dict(buf_spatial[var])
        cost[var] = iters[var] * data_size[var] * var_bits[var] * len(addr)
        cost[var + '_milp'] = data_size[var] * var_bits[var] * len(addr)
        cost[var + '_milp_spatial'] = data_size[var] * var_bits[var]
        logger.info("\t\t{}: iter={}, data_size={}, var_bits={}, unicast_count={}, cost={}".format(var, iters[var],
                                                                                                   data_size[var],
                                                                                                   var_bits[var],
                                                                                                   len(addr),
                                                                                                   cost[var]))
        total_cost += cost[var]

    var = 'Outputs_Store'
    addr = construct_addrs_dict(buf_spatial['Outputs'])
    cost[var] = iters[var] * data_size['Outputs'] * var_bits['Outputs'] * len(addr)
    cost[var + '_milp'] = data_size['Outputs'] * var_bits['Outputs'] * len(addr)
    logger.info("\t {}: iter={}, data_size={}, var_bits={}, unicast_count={}, cost={}".format(var, iters[var],
                                                                                              data_size['Outputs'],
                                                                                              var_bits['Outputs'],
                                                                                              len(addr), cost[var]))
    total_cost += cost[var]
    cost['Total'] = total_cost
    return cost


def new_generate_temp(temporal_loops, iter_start_dim, partial_sum_dep_prob_idx):
    loop_bound_tup = tuple([int(loop['end']) for loop in temporal_loops])
    loop_level = len(temporal_loops)

    iters = {}
    steps = {}
    for var in ['Weights', 'Inputs', 'Outputs', 'Outputs_Store']:
        steps[var] = np.zeros(loop_bound_tup, dtype=np.uint8)

    for var in ['Weights', 'Inputs']:
        start_dim = iter_start_dim[var]
        np_idx = []

        for dim in range(loop_level):
            if dim < start_dim:
                np_idx.append(0)
            else:
                np_idx.append(slice(None))
        np_idx_tup = tuple(np_idx)
        steps[var][np_idx_tup] = 1
        iters[var] = np.sum(steps[var])

    var = 'Outputs'
    if partial_sum_dep_prob_idx < loop_level:
        start_dim = iter_start_dim[var]
        np_idx = []
        for dim in range(loop_level):
            if dim < start_dim:
                np_idx.append(0)
            elif dim == partial_sum_dep_prob_idx:
                np_idx.append(slice(1, None))
            else:
                np_idx.append(slice(None))
        np_idx_tup = tuple(np_idx)
        steps[var][np_idx_tup] = 1
    iters[var] = np.sum(steps[var])

    var = 'Outputs_Store'
    start_dim = iter_start_dim['Outputs']
    np_idx = []
    for dim in range(loop_level):
        if dim < start_dim:
            np_idx.append(-1)
        else:
            np_idx.append(slice(None))
    np_idx_tup = tuple(np_idx)
    steps[var][np_idx_tup] = 1
    iters[var] = np.sum(steps[var])

    for var in ['Weights', 'Inputs', 'Outputs', 'Outputs_Store']:
        steps[var] = list(steps[var].transpose().flatten())
    # temporal_loop_iter = np.unravel_index(idx, loop_bound)
    return (steps, iters)


if __name__ == "__main__":
    module_name = os.path.basename(__file__).replace(".py", "")
    utils.setup_logging(module_name, logger)
    xml_file = "timeloop-model.map+stats.xml"
    subnest_info = get_subnest_info(xml_file);
    print(subnest_info)
    schedule = gen_schedule(subnest_info, start_level=4, end_level=5)
