import logging
import pathlib
import shutil

import cosa.utils
from cosa.parse_workload import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # capture everything


def run_config(mapspace, spatial_config, perm_config, factor_config, status_dict=dict(), run_gen_map=True,
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
            logging.info("status_dict: {}".format(status_dict[status_dict_key]))
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
    # print(output_dir)

    status_dict[status_dict_key]['output_dir'] = str(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = 'tc'

    map_path = output_dir / 'map_16.yaml'
    logging.info("map_path: {}".format(map_path))
    xml_file = output_dir / 'timeloop-model.map+stats.xml'
    csv_file = output_dir / "{}.csv".format(prefix)
    json_file = output_dir / "{}.json".format(prefix)
    stats_file = output_dir / "{}.summary.json".format(prefix)
    status_dict_file = output_dir / "{}.dict.json".format(prefix)

    # logging.debug("status_dict_before: {}".format(status_dict[status_dict_key]))
    # generate map 
    if run_gen_map:
        # print('run_timeloop> timeloop-model {} {} {}'.format(mapspace.arch.path, mapspace.prob.path, map_path))
        if map_path.exists() and not xml_file.exists():
            status_dict[status_dict_key]['run_status'][0] = 0
        elif not xml_file.exists():
            # logging.info("Run Generate Mapping")
            utils.store_yaml(map_path, mapping)
            success = utils.run_timeloop(mapspace.arch.path, mapspace.prob.path, map_path, cwd=output_dir)

            # if passes timeloop check
            if not success:
                status_dict[status_dict_key]['run_status'][0] = 0
                logging.info("\tInvalid Mapping Detected!")
                if delete_invalid:
                    shutil.rmtree(
                        output_base / mapspace.arch.config_str() / mapspace.prob.config_str() / mapspace.config_str()[
                            0])
                return status_dict[status_dict_key]
            else:
                # logging.info("\tValid Mapping Detected!")
                assert (xml_file.exists())
                status_dict[status_dict_key]['run_status'][0] = 1
        else:
            status_dict[status_dict_key]['run_status'][0] = 1

        # Parse buffer util
        if xml_file.exists():
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
    logging.info("Status: {}".format(status_dict[status_dict_key]))
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
