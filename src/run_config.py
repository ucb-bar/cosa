import logging
import pathlib
import shutil
from functools import reduce

import utils
from parse_workload import *

logger = logging.getLogger(__name__)
#logger.setLevel(logger.NOTSET)  # capture everything
# logger.disabled = True


def run_config(mapspace, spatial_config, perm_config, factor_config, status_dict=dict(), run_gen_map=True,
               run_gen_tc=False, run_sim_test=False, output_path='output_dir', spatial_configs=None, exe='sim_test',
               valid_check=False, nb_sim=False, outer_loopcount_limit=256, delete_invalid=False):
    logger.debug("factor_config: {}".format(factor_config))
    mapspace.update_mapspace(perm_config, factor_config)

    valid, invalid_mem_entries, invalid_mem_instances, utilized_mem_entries, utilized_mem_instances = mapspace.valid_check()
    while (not valid):
        mapspace.reset_mapspace(spatial_config, spatial_configs)
        update_greedy_factor_config(mapspace, invalid_mem_entries, invalid_mem_instances, factor_config)
        logger.debug("--------------")
        logger.debug("factor_config: {}".format(factor_config))
        logger.debug("factor_space_tup: {}".format(mapspace.factor_space_tup))
        mapspace.update_mapspace(perm_config, factor_config)
        valid, invalid_mem_entries, invalid_mem_instances, utilized_mem_entries, utilized_mem_instances = mapspace.valid_check()

    if valid:
        status_dict = run_simulation(mapspace, spatial_config, perm_config, factor_config, status_dict,
                                     run_gen_map=run_gen_map, run_gen_tc=run_gen_tc, run_sim_test=run_sim_test,
                                     output_path=output_path, spatial_configs=spatial_configs)

        logger.debug("utilized_mem_entries {}".format(utilized_mem_entries))
        utilized_mem_entries_prod = reduce(lambda x, y: float(x) * float(y), utilized_mem_entries[0:-1])
        utilized_mem_entries_sum = reduce(lambda x, y: x + y, utilized_mem_entries[0:-1])
        logger.debug(
            "utilized_mem_entries sum={}, prod={:e}".format(utilized_mem_entries_sum, utilized_mem_entries_prod))

        logger.debug("timeloop utilized_capacity {}".format(status_dict['utilized_capacity']))
        for idx, utilized_capacity in enumerate(utilized_mem_entries):
            if idx == len(utilized_mem_entries) - 1:
                continue
            try:
                assert (utilized_capacity == status_dict['utilized_capacity'][idx])
            except:
                logger.error("utilized_capacity {}".format(utilized_capacity))
                logger.error("timeloop utilized_capacity {}".format(status_dict['utilized_capacity']))
                raise RuntimeError('Schedule passed CoSA valid check but failed Timeloop simulation.')
    else:
        logger.debug("utilized_mem_entries {}".format(utilized_mem_entries))
    return status_dict


def run_simulation(mapspace, spatial_config, perm_config, factor_config, status_dict=dict(), run_gen_map=True,
                   run_gen_tc=False, run_sim_test=False, output_path='output_dir', spatial_configs=None, exe='sim_test',
                   valid_check=False, nb_sim=False, outer_loopcount_limit=256, delete_invalid=False):
    mapspace.reset_mapspace(spatial_config, spatial_configs)
    mapspace.update_mapspace(perm_config, factor_config)

    # if it does not pass the check skip it
    if valid_check:
        valid_mapping, _, _, _, _ = mapspace.valid_check()
        if not valid_mapping:
            status_dict = {'run_status': [0]}
            return status_dict
            # raise ValueError()

    # get the config_space_str as key to the status_dict
    # status_dict_key = mapspace.config_space_str(spatial_config, perm_config, factor_config)
    key_strs = [mapspace.arch.config_str(), mapspace.prob.config_str(), mapspace.config_str()[0],
                mapspace.config_str()[1]]

    status_dict_key = '+'.join(key_strs)
    status_dict_val = status_dict.get(status_dict_key)

    if status_dict_val:
        finish_run = True
        if run_gen_map:
            if status_dict_val['run_status'][0] == -1:
                finish_run = False
        if run_gen_tc:
            if status_dict_val['run_status'][1] == -1:
                finish_run = False
        if run_sim_test:
            if status_dict_val['run_status'][2] == -1:
                finish_run = False
        if finish_run:
            logger.info("status_dict: {}".format(status_dict[status_dict_key]))
            # return
            return status_dict_val
    else:
        idx = 0
        # idx can go out of index
        # idx = mapspace.get_idx_from_factor_config(factor_config)
        status_dict_val = {}
        status_dict_val['run_status'] = [-1, -1, -1]
        status_dict_val['cycle_results'] = [-1] * 6  # un-initialized
        status_dict_val['utilized_capacity'] = []
        status_dict[status_dict_key] = status_dict_val

    mapping = mapspace.generate_mapping()

    output_base = pathlib.Path(output_path).resolve()
    output_dir = output_base / mapspace.arch.config_str() / mapspace.prob.config_str() / mapspace.config_str()[0] / \
                 mapspace.config_str()[1]

    status_dict[status_dict_key]['output_dir'] = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = 'tc'

    map_path = output_dir / 'map_16.yaml'
    logger.info("map_path: {}".format(map_path))
    xml_file = output_dir / 'timeloop-model.map+stats.xml'
    stats_txt_file = output_dir / 'timeloop-model.stats.txt'
    csv_file = output_dir / "{}.csv".format(prefix)
    json_file = output_dir / "{}.json".format(prefix)
    stats_file = output_dir / "{}.summary.json".format(prefix)
    status_dict_file = output_dir / "{}.dict.json".format(prefix)

    # logger.debug("status_dict_before: {}".format(status_dict[status_dict_key]))
    # generate map 
    if run_gen_map:
        if map_path.exists() and not stats_txt_file.exists():
            status_dict[status_dict_key]['run_status'][0] = 0
        elif not stats_txt_file.exists():
            # logger.info("Run Generate Mapping")
            utils.store_yaml(map_path, mapping)
            success = utils.run_timeloop(mapspace.arch.path, mapspace.prob.path, map_path, cwd=output_dir)

            # if passes timeloop check
            if not success:
                status_dict[status_dict_key]['run_status'][0] = 0
                logger.info("\tInvalid Mapping Detected!")
                if delete_invalid:
                    shutil.rmtree(
                        output_base / mapspace.arch.config_str() / mapspace.prob.config_str() / mapspace.config_str()[
                            0])
                return status_dict[status_dict_key]
            else:
                # logger.info("\tValid Mapping Detected!")
                assert (xml_file.exists())
                status_dict[status_dict_key]['run_status'][0] = 1
        else:
            status_dict[status_dict_key]['run_status'][0] = 1

        # Parse buffer util
        if stats_txt_file.exists():
            subnest_info = get_subnest_info(xml_file)
            bufsize = subnest_info['bufsize']
            pe_cycle = subnest_info['pe_cycle']
            pe_energy = subnest_info['pe_energy']
            cycle = subnest_info['cycle']
            energy = subnest_info['energy']
            status_dict[status_dict_key]['pe_cycle'] = pe_cycle
            status_dict[status_dict_key]['pe_energy'] = pe_energy
            status_dict[status_dict_key]['energy'] = energy
            status_dict[status_dict_key]['cycle'] = cycle
            for buf_idx, (buf_name, buf) in enumerate(bufsize.items()):
                utilized_capacity = 0
                for mem_util in buf:
                    utilized_capacity += mem_util
                status_dict[status_dict_key]['utilized_capacity'].append(utilized_capacity)
            # parse area
            area = utils.get_area_stats(stats_txt_file)
            status_dict[status_dict_key]['area'] = area

    logger.info("Status: {}".format(status_dict[status_dict_key]))
    utils.store_json(status_dict_file, status_dict, indent=4)
    return status_dict[status_dict_key]


def get_perm_size(mapspace, spatial_config, perm_config, factor_config, status_dict, output_path='output_dir',
                  spatial_configs=None, exe='sim_test', valid_check=False, outer_loopcount_limit=None):
    status_dict = run_config(mapspace, spatial_config, perm_config, factor_config, status_dict, run_gen_map=True,
                             run_gen_tc=True, run_sim_test=False, output_path=output_path,
                             spatial_configs=spatial_configs, exe=exe, valid_check=valid_check,
                             outer_loopcount_limit=outer_loopcount_limit)
    sizes = []
    for var in ['Weights', 'Inputs', 'Outputs']:
        sizes.append(status_dict['cost']['{}_milp'.format(var)])
    return sizes


def get_spatial_size(mapspace, spatial_config, perm_config, factor_config, status_dict, output_path='output_dir',
                     spatial_configs=None, exe='sim_test', valid_check=False, outer_loopcount_limit=None):
    status_dict = run_config(mapspace, spatial_config, perm_config, factor_config, status_dict, run_gen_map=True,
                             run_gen_tc=True, run_sim_test=False, output_path=output_path,
                             spatial_configs=spatial_configs, exe=exe, valid_check=valid_check,
                             outer_loopcount_limit=outer_loopcount_limit)
    sizes = []
    for var in ['Weights', 'Inputs', 'Outputs']:
        sizes.append(status_dict['cost']['{}_milp_spatial'.format(var)])
    return sizes


def update_greedy_factor_config(mapspace, invalid_mem_entries, invalid_mem_instances, factor_config, move_outter=True):
    # first resolve invalid_mem_instances
    for mem_idx in invalid_mem_instances:
        # min_idx = None
        # min_mem_idx = mapspace.total_factor_choices - 1
        for prob_idx, (prob_factor_configs, prob_factors) in enumerate(
                zip(factor_config, mapspace.unschduled_prob_factors)):
            if (len(prob_factors) != 1 or prob_factors[0] != 1):
                # if prob idx is within the mem 
                for i, factor_mem_idx in enumerate(prob_factor_configs):
                    if factor_mem_idx >= mapspace.arch.mem_levels:
                        valid_spatial_mem_idx = factor_mem_idx - mapspace.arch.mem_levels
                        spatial_mem_idx, _ = mapspace.valid_spatial_levels[valid_spatial_mem_idx]

                        if mem_idx <= spatial_mem_idx:
                            if factor_config[prob_idx][i] - 1 >= mapspace.arch.mem_levels:
                                factor_config[prob_idx][i] -= 1
                            else:
                                factor_config[prob_idx][i] = 1
                            return

    for mem_idx in invalid_mem_entries:
        bound = len(factor_config)
        prob_idxs = []
        factor_mem_idxs = []
        spatial_factor_mem_idxs = []
        spatial_prob_idxs = []
        for prob_idx in range(bound - 1, -1, -1):
            prob_factor_configs = factor_config[prob_idx]
            prob_factors = mapspace.unschduled_prob_factors[prob_idx]
            # for prob_idx, (prob_factor_configs, prob_factors) in enumerate(zip(factor_config, mapspace.unschduled_prob_factors)):
            # if mem_idx in mapspace.prob_mem[prob_idx]:
            if mapspace.is_mem_related_to_prob(mem_idx, prob_idx):
                if (len(prob_factors) != 1 or prob_factors[0] != 1):
                    # if prob idx is within the mem 
                    for i, factor_mem_idx in enumerate(prob_factor_configs):
                        # if mapped to temperal 
                        if factor_mem_idx < mapspace.arch.mem_levels:
                            if factor_mem_idx <= mem_idx:
                                if not move_outter:
                                    factor_config[prob_idx][i] = mem_idx + 1
                                    return
                                else:
                                    prob_idxs.append((prob_idx, i))
                                    factor_mem_idxs.append(factor_mem_idx)
                        else: # spatial
                            if factor_mem_idx - mapspace.arch.mem_levels <= mem_idx:
                                spatial_prob_idxs.append((prob_idx, i))
                                spatial_factor_mem_idxs.append(factor_mem_idx)


        if move_outter:
            # select one value, must exist 
            if len(factor_mem_idxs) > 0:
                max_mem_idx = factor_mem_idxs[0]
                select_prob_idx = prob_idxs[0]
                for i, factor_mem_idx in enumerate(factor_mem_idxs):
                    if factor_mem_idx > max_mem_idx:
                        max_mem_idx = factor_mem_idx
                        select_prob_idx = prob_idxs[i]
                factor_config[select_prob_idx[0]][select_prob_idx[1]] = mem_idx + 1
            elif len(spatial_factor_mem_idxs) > 0:
                max_mem_idx = spatial_factor_mem_idxs[0]
                select_prob_idx = spatial_prob_idxs[0]
                # find max spatial mem idx
                for i, factor_mem_idx in enumerate(spatial_factor_mem_idxs):
                    if factor_mem_idx > max_mem_idx:
                        max_mem_idx = factor_mem_idx
                        select_prob_idx = spatial_prob_idxs[i]
                spatial_mem_idx = mapspace.valid_spatial_levels[max_mem_idx - mapspace.arch.mem_levels][0]
                factor_config[select_prob_idx[0]][select_prob_idx[1]] = spatial_mem_idx + 1
            else:
                raise ValueError('Invalid mapping.')
            return
