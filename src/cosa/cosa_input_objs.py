#!/usr/bin/env python3 
import copy
import logging
import pathlib

import numpy as np
import cosa.run_config

import cosa.utils as utils
from cosa.parse_workload import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # capture everything


class Prob(object):
    """Problem space with layer dimension, stride and dilation defined.
    
    Attributes: 
        prob: A layer dimemsion dictionary. 
            R, S represent the weight filter width and height.
            P, Q represent the output feature map width and height.
            C represents the input channel size. 
            K represents the output channel size.
            N represents the batch size. 
            Wstride, Hstride represent the width and height dimension stride.
            Wdilation, Hdilation represent the width and height dimension dilation.
        prob_bound:  A 1d array with layer dimension value for R,S,P,Q,C,K,N
            e.g. [1,1,1,2,3,4,5]
        prob_factors:  A 2d array with all prime factors generated from each dimension
            e.g. [[1],[1],[1],[2],[3],[2,2],[5]] 
    """

    def __init__(self, prob_path):
        """Initialize the layer dimension from an input yaml file. 

            Example input yaml file format: 
                problem:
                  C: 3
                  Hdilation: 1
                  Hstride: 2
                  K: 64
                  N: 1
                  P: 112
                  Q: 112
                  R: 7
                  S: 7
                  Wdilation: 1
                  Wstride: 2
                  shape: cnn-layer


        Args: 
            prob_path: Path to the yaml file that defines the convolution layer dimensions. 
        """
        # defines the dimension index for 7 major loop bounds 
        self.prob_idx_name_dict = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'N'}
        self.prob_name_idx_dict = {v: k for k, v in self.prob_idx_name_dict.items()}

        self.prob_bound = [-1] * len(self.prob_name_idx_dict)
        self.prob_factors = []
        for i in range(len(self.prob_name_idx_dict)):
            self.prob_factors.append([])

        self.prob_levels = len(self.prob_idx_name_dict.items())

        self.path = prob_path.resolve()
        prob_dict = utils.parse_yaml(self.path)
        self.prob = prob_dict['problem']

        for key, value in self.prob.items():
            if ('stride' in key or 'dilation' in key):
                continue
            if (key == 'shape'):
                continue
            prob_idx = self.prob_name_idx_dict[key]
            self.prob_bound[prob_idx] = value
            self.prob_factors[prob_idx] = utils.get_prime_factors(value)

    def config_str(self):
        """Returnsthe key str name for representing a unique layer."""
        val_arr = []
        for value in self.prob_bound:
            val_arr.append(str(value))
        keys = ['Wstride', 'Hstride', 'Wdilation', 'Hdilation']
        val_arr.extend([str(self.prob[key]) for key in keys])
        val_str = "_".join(val_arr)
        return val_str

    def print(self):
        print(self.__dict__)


class Arch(object):
    """ Hardware architecture specifyng number of hardware instances and buffer capacity.
    
    Attributes: 
        mem_instances: number of memory instances per chip.
        mem_entries: number of valid memory entries.
    """

    def __init__(self, arch_path):
        """Initialize the hardware architecture details from an input yaml file. 

        Args: 
            arch_path: Path to the yaml file that defines the hardware architecture constraints. 
        """

        self.path = arch_path.resolve()
        arch_dict = utils.parse_yaml(self.path)

        # arch config version, please add a postfix _v3 to
        # the yaml filename if a new version is used
        version = 'v3' if '_v3' in self.path.name else 'v1'

        # mem instance size for each 
        self.mem_instances = []
        self.mem_entries = []

        # name to idx lookup
        self.mem_idx = {}

        # idx to name lookup
        self.mem_name = {}

        if version == 'v1':
            self.arch = arch_dict['arch']
            for key, value in self.arch.items():
                setattr(self, key, value)
            for i, mem in enumerate(self.storage):
                self.mem_idx[mem['name']] = i
                self.mem_name[i] = mem['name']
                self.mem_instances.append(mem['instances'])
                if i < len(self.storage) - 1:
                    self.mem_entries.append(mem['entries'])
        elif version == 'v3':
            self.dram = arch_dict['architecture']['subtree'][0]['local'][0]
            self.global_buf = arch_dict['architecture']['subtree'][0]['subtree'][0]['local'][0]
            self.pe_buf = arch_dict['architecture']['subtree'][0]['subtree'][0]['subtree'][0]['local']
            idx = 0
            for i, mem in enumerate(self.pe_buf[::-1]):
                if mem['class'] == 'SRAM' or mem['class'] == 'regfile':
                    self.mem_idx[mem['name']] = idx
                    self.mem_name[idx] = mem['name']
                    self.mem_instances.append(mem['attributes']['instances'])
                    self.mem_entries.append(mem['attributes']['entries'])
                    idx += 1

            self.mem_idx[self.global_buf['name']] = idx
            self.mem_name[idx] = self.global_buf['name']
            self.mem_instances.append(self.global_buf['attributes']['instances'])
            self.mem_entries.append(self.global_buf['attributes']['entries'])
            idx += 1
            self.mem_idx[self.dram['name']] = idx
            self.mem_name[idx] = self.dram['name']
            self.mem_instances.append(self.dram['attributes']['instances'])
            self.arch = {"instances": self.mem_instances, "entries": self.mem_entries}
        self.mem_levels = len(self.mem_idx.items())
        self.S = self.gen_spatial_constraint()

    def gen_spatial_constraint(self):
        """Generate spatial constraints."""
        S = []
        inner_instances = self.mem_instances[0]
        for i in self.mem_instances:
            if i != 0:
                S.append(inner_instances // i)
                inner_instances = i
        return S

    def config_str(self):
        """Return the filename for the input yaml with postfix."""
        return self.path.stem

    def print(self):
        print(self.__dict__)


class Mapspace(object):
    """ Mapping Space"""

    def __init__(self, path):
        mapspace_dict = utils.parse_yaml(path)
        self.mapspace = mapspace_dict['mapspace']

        for key, value in self.mapspace.items():
            setattr(self, key, value)

        self.var_idx_dict = {'Weights': 0, 'Inputs': 1, 'Outputs': 2}
        self.var_name_dict = {v: k for k, v in self.var_idx_dict.items()}
        self.org_idx_dict = {'spatial': 0, 'temporal': 1}
        self.org_name_dict = {v: k for k, v in self.org_idx_dict.items()}
        self.config_idx_dict = {'perm': 0, 'factor': 1}
        self.mapspace = None
        self.bypass = None
        self.arch = None
        self.prob = None
        self.factor_space = None
        self.factor_config_tup = None
        self.perm_space = None
        self.spatial_space = None

    def init(self, prob, arch, use_valid_spatial_levels=True):
        self.prob = prob
        self.arch = arch

        # init bypass 
        # order for the spatial dim indicates XY spatial arrangement if X!=Y
        self.bypass = np.zeros((arch.mem_levels, len(self.var_idx_dict.items())), dtype=np.int8)

        # parse bypass info
        for constraint in self.constraints:
            if constraint['type'] == 'datatype':
                mem_idx = arch.mem_idx[constraint['target']]
                bypass_vars = constraint['bypass']
                if bypass_vars is not None:
                    for var in bypass_vars:
                        var_idx = self.var_idx_dict[var]
                        self.bypass[mem_idx][var_idx] = 1

        # init valid spatial level from arch
        self.valid_spatial_levels = self.get_valid_spatial_levels(use_valid_spatial_levels)
        self.valid_spatial_factors = [1] * arch.mem_levels
        for mem_level, valid_spatial_factors in self.valid_spatial_levels:
            self.valid_spatial_factors[mem_level] = valid_spatial_factors

        self.prob_var_dict = {
            0: ["Weights"],
            1: ["Weights"],
            2: ["Inputs", "Outputs"],
            3: ["Inputs", "Outputs"],
            4: ["Inputs", "Weights"],
            5: ["Outputs", "Weights"],
            6: ["Inputs", "Outputs"],
        }

        # parse memory-related variables 
        self.var_mem_dict = self.get_var_mem()

        # init
        spatial_configs = [1] * self.prob.prob_levels
        self._reset_mapspace(spatial_configs)

        # init the ordering
        # inner to outer 
        # idx of the prob dimension
        perm_config = self.get_default_perm()

        # set up permutation space 
        self.perm_space = np.math.factorial(self.prob.prob_levels) * np.ones(
            (arch.mem_levels, len(self.org_idx_dict.items())))
        self.perm_space_tup = tuple(self.perm_space.flatten())

        self.perm_arr_tup = tuple(range(self.prob.prob_levels, 0, -1))

        # 7 x prob_factors
        factor_config = self.get_default_factor()
        self.update_mapspace(perm_config, factor_config)

    # input 333 -> order_idx [0, 2, 3, 3, 1, 1, 0]
    # order_idx -> perm_arr [0, 3, 5, 6, 2, 4, 1]
    def get_perm_arr_from_val(self, val):
        perm_arr = utils.get_perm_arr_from_val(val, self.perm_arr_tup, self.prob.prob_levels)
        return perm_arr

    # input perm_arr [0, 3, 5, 6, 2, 4, 1] -> order_idx 
    # order_idx -> val 333 
    def get_val_from_perm_arr(self, perm_arr):
        val = utils.get_val_from_perm_arr(self, perm_arr, self.perm_arr_tup, self.prob.prob_levels)
        return val

    def get_perm_config_from_idx(self, idx):
        perm_idx_arr = np.unravel_index(idx, self.perm_config_tup)
        perm_idx = perm_idx_arr.reshape(self.perm_config)
        return perm_idx

    def get_idx_from_perm_config(self, perm_config):
        perm_config_tup = tuple(perm_config.flatten())
        idx = np.ravel_multi_index(perm_config_tup, self.perm_space_tup)
        return idx

    # get a pruned space, instead of 7 mem level just have 6 excluding the Register
    def get_factor_config_from_prune_idx(self, idx):
        factor_space = list(self.factor_space_tup)
        factor_space = [factor - 1 if factor > 1 else factor for factor in factor_space]
        factor_idx_arr = np.unravel_index(idx, tuple(factor_space))

        factor_config = copy.deepcopy(self.factor_space)
        idx = 0
        for i, sublist in enumerate(factor_config):
            for j, item in enumerate(sublist):
                val = factor_idx_arr[idx]
                if self.factor_space[i][j] > 1:
                    val += 1
                factor_config[i][j] = val
                idx += 1
        return factor_config

    def get_factor_config_from_idx(self, idx):
        factor_idx_arr = np.unravel_index(idx, self.factor_space_tup)
        factor_config = copy.deepcopy(self.factor_space)
        idx = 0
        for i, sublist in enumerate(factor_config):
            for j, item in enumerate(sublist):
                factor_config[i][j] = factor_idx_arr[idx]
                idx += 1
        return factor_config

    def get_idx_from_factor_config(self, factor_config):
        factor_config_tup = tuple([item for sublist in factor_config for item in sublist])
        idx = np.ravel_multi_index(factor_config_tup, self.factor_space_tup)
        return idx

    def get_default_factor(self):
        factor_config = []
        for prob_factor in self.unschduled_prob_factors:
            # init factor to temporal outer most loop 
            factors = [self.arch.mem_levels - 1] * len(prob_factor)
            factor_config.append(factors)
        return factor_config

    def get_default_perm(self):
        perm_config = []
        for mem_idx in range(self.arch.mem_levels):
            # for org_idx in range(self.mapspace.shape[1]):
            permutation = []
            for prob_idx in range(self.prob.prob_levels):
                permutation.append(prob_idx)
            perm_config.append(permutation)
        return perm_config

        # spatial_config is to make it backforward compatible

    # if spatial_configs = '0_0_0_0_0_0_0', no preset spatial dim for the NoC
    def reset_mapspace(self, spatial_config, spatial_configs=None):
        if spatial_configs is None:
            spatial_configs = [1] * self.prob.prob_levels
            prob_idx = self.prob.prob_name_idx_dict[spatial_config]  # C-4 K-5
            spatial_configs[prob_idx] = 16
        self._reset_mapspace(spatial_configs)

    def _reset_mapspace(self, spatial_configs):
        # reset mapspace
        self.mapspace = np.ones((self.arch.mem_levels, len(self.org_idx_dict.items()),
                                 self.prob.prob_levels, len(self.config_idx_dict.items())), dtype=np.int16)

        # reset copy of prob_factors  
        self.unschduled_prob_factors = copy.deepcopy(self.prob.prob_factors)

        # TODO load the maspace contraints to mapspace
        # init the spatial PE
        spatial_mem_idx = self.mapspace.shape[0] - 1
        for i in range(len(self.arch.mem_instances) - 1, -1, -1):
            if self.arch.mem_instances[i] == 1:
                spatial_mem_idx = i
        # only spatial partition at DRAM  
        org_idx = 0

        is_spatial_specified = False
        for prob_idx, spatial_factor in enumerate(spatial_configs):
            if spatial_factor > 0:
                self.mapspace[spatial_mem_idx][org_idx][prob_idx][1] = spatial_factor

                utils.update_prime_factors(self.unschduled_prob_factors, prob_idx, spatial_factor)
                is_spatial_specified = True

        # for each factor choose a memory level to be in
        # TODO not considering the spatial probabiliy
        # including spatial and temporal dims 
        # spatial - 1 for the NoC cores 
        # TODO make a control signal to enable or disable heuristic spatial level
        num_valid_spatial_levels = len(self.valid_spatial_levels)
        if is_spatial_specified:
            num_valid_spatial_levels = num_valid_spatial_levels - 1

        self.total_factor_choices = self.arch.mem_levels + num_valid_spatial_levels

        self.factor_space = []
        for prob_factor in self.unschduled_prob_factors:
            # set the bound to one, if the factor is 1
            if len(prob_factor) == 1 and prob_factor[0] == 1:
                bounds = [1]
            else:
                bounds = [self.total_factor_choices] * len(prob_factor)
            self.factor_space.append(bounds)
        self.factor_space_tup = tuple([item for sublist in self.factor_space for item in sublist])

    def update_mapspace(self, perm_config, factor_config):
        # update factors
        for prob_idx, mem_levels in enumerate(factor_config):
            for factor_idx, mem_idx in enumerate(mem_levels):
                if mem_idx < self.arch.mem_levels:
                    self.mapspace[mem_idx, 1, prob_idx, 1] *= self.unschduled_prob_factors[prob_idx][factor_idx]
                else:
                    spatial_mem_idx = self.valid_spatial_levels[mem_idx - self.arch.mem_levels][0]
                    self.mapspace[spatial_mem_idx, 0, prob_idx, 1] *= self.unschduled_prob_factors[prob_idx][factor_idx]
        # update permutation
        for mem_idx, permutations in enumerate(perm_config):
            for prob_idx, order in enumerate(permutations):
                self.mapspace[mem_idx, :, prob_idx, 0] = order

    def config_feature(self, perm_config, factor_config, org_idx=1):
        features = np.zeros((self.arch.mem_levels, self.prob.prob_levels), np.int16)
        for mem_idx in range(self.mapspace.shape[0]):
            for prob_idx in range(self.mapspace.shape[2]):
                factor = self.mapspace[mem_idx][org_idx][prob_idx][1]
                perm_idx = self.mapspace[mem_idx][org_idx][prob_idx][0]
                features[mem_idx][perm_idx] = factor
        logger.debug("features {}".format(features))
        return features

    def config_space_str(self, spatial_config, perm_config, factor_config):
        prob_factor_strs = []
        for mem_levels in factor_config:
            # factors = [self.arch.mem_levels-1] * len(prob_factor)
            # self.factor_config.append(factors)
            mem_level_arr = [str(mem_level) for mem_level in mem_levels]
            mem_level_str = "_".join(mem_level_arr)
            prob_factor_strs.append(mem_level_str)
        prob_factor_str = "-".join(prob_factor_strs)
        prob_perm_strs = []
        for mem_idx, permutations in enumerate(perm_config):
            perm_arr = [str(perm) for perm in permutations]
            perm_str = "".join(perm_arr)
            prob_perm_strs.append(perm_str)
        prob_perm_str = "-".join(prob_perm_strs)

        config_str = spatial_config + '+' + prob_perm_str + '+' + prob_factor_str
        # assert(len(config_str) < 256)
        return config_str

    def config_factor_str(self, org_idx):
        factor_strs = []
        for mem_idx in range(self.mapspace.shape[0]):
            factor_arr = []
            for prob_idx in range(self.mapspace.shape[2]):
                factor_arr.append(str(self.mapspace[mem_idx][org_idx][prob_idx][1]))
            factor_str = '_'.join(factor_arr)
            factor_strs.append(factor_str)
        config_factor_str = "-".join(factor_strs)
        return config_factor_str

    def config_perm_str(self, org_idx):
        perm_strs = []
        for mem_idx in range(self.mapspace.shape[0]):
            perm_arr = []
            for prob_idx in range(self.mapspace.shape[2]):
                perm_arr.append(str(self.mapspace[mem_idx][org_idx][prob_idx][0]))
            perm_str = "".join(perm_arr)
            perm_strs.append(perm_str)
        config_perm_str = "-".join(perm_strs)
        return config_perm_str

    def config_str(self):
        config_str = []
        for org_idx in range(self.mapspace.shape[1]):
            factor_strs = []
            config_factor_str = self.config_factor_str(org_idx)
            config_perm_str = self.config_perm_str(org_idx)
            config_str.append(config_factor_str + '+' + config_perm_str)
        return config_str

        # TODO Write fast valid mapping checker, instead of running timeloop every time

    def is_valid_mapping(self, factor_config, factor_space):
        # there should not be any factor allocated at the  the register level  
        for prob_idx, mem_levels in enumerate(factor_config):
            for mem_level, _ in enumerate(mem_levels):
                if (factor_config[prob_idx][mem_level] == 0) and (factor_space[prob_idx][mem_level] != 1):
                    return False
        return True

    # return valid spatial factor bound from the arch
    def get_valid_spatial_levels(self, use_valid_spatial_levels):
        valid_spatial_levels = []
        # for i, mem in enumerate(self.arch.storage):
        for i, instances in enumerate(self.arch.mem_instances):
            if i == 0:
                if not use_valid_spatial_levels:
                    valid_spatial_levels.append((i, 1))
                continue
            # inner_instances = self.arch.storage[i-1]['instances']
            inner_instances = self.arch.mem_instances[i - 1]
            # cur_instances = mem['instances']
            cur_instances = self.arch.mem_instances[i]
            assert (inner_instances % cur_instances == 0)
            spatial_factor = inner_instances // cur_instances
            if use_valid_spatial_levels:
                if spatial_factor != 1:
                    valid_spatial_levels.append((i, spatial_factor))
            else:
                valid_spatial_levels.append((i, spatial_factor))
        return valid_spatial_levels

    def random_factor_config(self):
        factor_config = []
        # for different prob dim, # factors choose a mem lepatial_levelsyel to be in 
        for prob_idx, mem_levels in enumerate(self.factor_space):
            factors = np.random.randint(self.total_factor_choices, size=len(mem_levels))
            factor_config.append(factors)
        return factor_config

    def generate_tile_str(self, factors):
        tile_arr = []
        for name, idx in self.prob.prob_name_idx_dict.items():
            tile_arr.append("{}={}".format(name, factors[idx]))
        tile_str = " ".join(tile_arr)
        return tile_str

    # permutations store the loop order of each var
    def generate_perm_str(self, permutations):
        perm_arr = [None] * len(permutations)
        for name, idx in self.prob.prob_name_idx_dict.items():
            perm_arr[permutations[idx]] = name

        # There should be no None if permutation is right 
        perm_str = "".join(perm_arr)
        return perm_str

    def get_input_related_mem(self):
        var_idx = self.var_idx_dict['Inputs']
        mems = []
        for mem_idx in range(self.arch.mem_levels):
            if self.bypass[mem_idx][var_idx] != 1:
                mems.append(mem_idx)
        return mems

    def get_mem_var(self):
        mem_var_dict = {}
        for mem_idx in range(self.arch.mem_levels):
            mem_var_dict[mem_idx] = []

        for mem_idx in range(self.arch.mem_levels):
            for k, v in self.var_idx_dict.items():
                if self.bypass[mem_idx][v] != 1:
                    mem_var_dict[mem_idx].append(k)
        return mem_var_dict

    def get_var_mem(self):
        var_mem_dict = {}
        for var in self.var_idx_dict.keys():
            var_mem_dict[var] = []

        for var, idx in self.var_idx_dict.items():
            for mem_idx in range(self.arch.mem_levels):
                if self.bypass[mem_idx][idx] != 1:
                    var_mem_dict[var].append(mem_idx)
        return var_mem_dict

    def is_mem_related_to_prob(self, mem_idx, prob_idx):
        related = False
        for var in self.prob_var_dict[prob_idx]:
            if mem_idx in self.var_mem_dict[var]:
                related = True
        return related

    def get_mem_util(self):
        var_size = len(self.var_idx_dict)
        utilized_mem_entries = np.ones((self.mapspace.shape[0], var_size), dtype=np.uint32)
        utilized_mem_prob_entries = np.ones((self.mapspace.shape[0], self.mapspace.shape[2], var_size), dtype=np.uint32)
        mem_prob_entries = np.ones((self.mapspace.shape[0], self.mapspace.shape[2]), dtype=np.uint32)
        # get utilized mem size, using both spatial and temporal dims
        for mem_idx in range(self.mapspace.shape[0]):
            for org_idx in range(self.mapspace.shape[1]):
                for prob_idx in range(self.mapspace.shape[2]):
                    # for each factor that is not zero
                    factor = self.mapspace[mem_idx][org_idx][prob_idx][1]
                    if factor > 1:
                        for var in self.prob_var_dict[prob_idx]:
                            var_idx = self.var_idx_dict[var]
                            for utilized_mem_idx in self.var_mem_dict[var]:
                                if utilized_mem_idx >= mem_idx:
                                    utilized_mem_entries[utilized_mem_idx][var_idx] *= factor
                                    utilized_mem_prob_entries[utilized_mem_idx][prob_idx][var_idx] *= factor

                        for utilized_mem_idx in range(self.mapspace.shape[0]):
                            if utilized_mem_idx >= mem_idx:
                                mem_prob_entries[utilized_mem_idx][prob_idx] *= factor
        logger.debug("utilized_mem_prob_entries: {}".format(utilized_mem_prob_entries))
        # For buffers that are related to P and Q we need to see if R and S is 
        # TODO hardcoded here. for the input buffer, or buffers stores Inputs from the bypass constraints 
        # I = (P - 1) * Stride + R  
        input_related_mems = self.get_input_related_mem()
        input_idx = self.var_idx_dict["Inputs"]
        for input_related_mem_idx in input_related_mems:
            # for dram assume no extra input needs to be loaded
            if utilized_mem_prob_entries[input_related_mem_idx][2][input_idx] > 1 or \
                    mem_prob_entries[input_related_mem_idx][0] > 1:
                utilized_mem_prob_entries[input_related_mem_idx][2][input_idx] = (mem_prob_entries[
                                                                                      input_related_mem_idx][2] - 1) * \
                                                                                 self.prob.prob['Wstride'] + \
                                                                                 mem_prob_entries[
                                                                                     input_related_mem_idx][0]
                strid = self.prob.prob['Wstride']
            if utilized_mem_prob_entries[input_related_mem_idx][3][input_idx] > 1 or \
                    mem_prob_entries[input_related_mem_idx][1] > 1:
                utilized_mem_prob_entries[input_related_mem_idx][3][input_idx] = (mem_prob_entries[
                                                                                      input_related_mem_idx][3] - 1) * \
                                                                                 self.prob.prob['Hstride'] + \
                                                                                 mem_prob_entries[
                                                                                     input_related_mem_idx][1]
            # Update entry factors 
            # logger.debug("utilized_mem_entries before {}".format(utilized_mem_entries[input_related_mem_idx][input_idx]))
            utilized_mem_entries[input_related_mem_idx][input_idx] = 1
            for prob_idx in range(self.mapspace.shape[2]):
                utilized_mem_entries[input_related_mem_idx][input_idx] *= \
                    utilized_mem_prob_entries[input_related_mem_idx][prob_idx][input_idx]
            # logger.debug("utilized_mem_entries after {}".format(utilized_mem_entries[input_related_mem_idx][input_idx]))

        # logger.debug("utilized_mem_prob_entries after: {}".format(utilized_mem_prob_entries))

        utilized_mem_instances = [1] * self.mapspace.shape[0]
        utilized_spatial_factors = [1] * self.mapspace.shape[0]

        prev_spatial_instances = 1
        # for instances, check spatial dimension only 
        for mem_idx in range(self.mapspace.shape[0] - 1, -1, -1):
            org_idx = 0
            utilized_mem_instances[mem_idx] = prev_spatial_instances
            for prob_idx in range(self.mapspace.shape[2]):
                # for each factor that is not zero
                factor = self.mapspace[mem_idx][org_idx][prob_idx][1]
                if factor > 1:
                    prev_spatial_instances *= factor
                    utilized_spatial_factors[mem_idx] *= factor

        logger.debug("utilized_mem_entries: {}".format(utilized_mem_entries))
        logger.debug("utilized_spatial_factors: {}".format(utilized_spatial_factors))
        return utilized_mem_entries, utilized_mem_instances, utilized_spatial_factors

    def valid_check(self):
        utilized_mem_entries, utilized_mem_instances, utilized_spatial_factors = self.get_mem_util()

        # sum up all var mem util size for each mem level 
        total_utilized_mem_entries = np.zeros(self.mapspace.shape[0], dtype=np.uint32)
        for mem_idx in range(self.arch.mem_levels - 1):
            for var_name, var_idx in self.var_idx_dict.items():
                if mem_idx in self.var_mem_dict[var_name]:
                    total_utilized_mem_entries[mem_idx] += utilized_mem_entries[mem_idx][var_idx]

        utilized_mem_entries = total_utilized_mem_entries
        logger.debug("summed utilized_mem_entries: {}".format(utilized_mem_entries))

        valid = True
        invalid_mem_entries = []
        invalid_mem_instances = []
        logger.debug("utilized_mem_instances: {}".format(utilized_mem_instances))
        logger.debug("mem_entries: {}".format(self.arch.mem_entries))
        logger.debug("mem_instances: {}".format(self.arch.mem_instances))
        logger.debug("valid_spatial_factors: {}".format(self.valid_spatial_factors))

        # check if total entries/instances exceeds the maximum size/instances, 
        # except for DRAM
        for i, mem_entries in enumerate(utilized_mem_entries):
            if utilized_mem_instances[i] > self.arch.mem_instances[i] or utilized_spatial_factors[i] > \
                    self.valid_spatial_factors[i]:
                valid = False
                invalid_mem_instances.append(i)
            if i < len(utilized_mem_entries) - 1 and mem_entries > self.arch.mem_entries[i]:
                valid = False
                invalid_mem_entries.append(i)

        logger.debug("invalid_mem_entries: {}".format(invalid_mem_entries))
        logger.debug("invalid_mem_instances: {}".format(invalid_mem_instances))
        return (valid, invalid_mem_entries, invalid_mem_instances, utilized_mem_entries, utilized_mem_instances)

    def generate_greedy_mapspace(self):
        mapping = {}
        mapping['mapping'] = []
        map_prob_bound = [1] * self.mapspace.shape[2]

        for mem_idx in range(self.mapspace.shape[0]):
            for org_idx in range(self.mapspace.shape[1]):
                factors = []
                permutations = []
                config = {}
                # No fanout to inside mem for the innermost mem
                if org_idx == 0:
                    # if mem_idx < self.mapspace.shape[0] - 1:
                    if mem_idx > 0:
                        if self.arch.storage[mem_idx]['instances'] == self.arch.storage[mem_idx - 1]['instances']:
                            continue
                    elif mem_idx == 0:
                        continue

                config['target'] = self.arch.mem_name[mem_idx]
                config['type'] = self.org_name_dict[org_idx]
                for prob_idx in range(self.mapspace.shape[2]):
                    permutations.append(self.mapspace[mem_idx][org_idx][prob_idx][0])
                    factors.append(self.mapspace[mem_idx][org_idx][prob_idx][1])
                    if self.mapspace[mem_idx][org_idx][prob_idx][1] != 1:
                        map_prob_bound[prob_idx] *= self.mapspace[mem_idx][org_idx][prob_idx][1]
                config['factors'] = self.generate_tile_str(factors)
                config['permutation'] = self.generate_perm_str(permutations)
                mapping['mapping'].append(config)

        # Make sure the prob size for the mapping matches the orig prob size
        try:
            assert (map_prob_bound == self.prob.prob_bound[0:len(map_prob_bound)])
        except:
            logger.error("{} not equals to {}".format(map_prob_bound, self.prob.prob_bound[0:len(map_prob_bound)]))

        for mem_idx in range(self.bypass.shape[0]):
            bypass = []
            keep = []
            config = {}
            config['target'] = self.arch.mem_name[mem_idx]
            config['type'] = 'bypass'
            for var_idx in range(self.bypass.shape[1]):
                if self.bypass[mem_idx][var_idx]:
                    bypass.append(self.var_name_dict[var_idx])
                else:
                    keep.append(self.var_name_dict[var_idx])
            config['bypass'] = bypass
            config['keep'] = keep
            mapping['mapping'].append(config)

        return mapping

    def generate_greedy_mapspace(self, template_dict):
        map_prob_bound = [1] * self.mapspace.shape[2]

        for mem_idx in range(self.mapspace.shape[0]):
            for org_idx in range(self.mapspace.shape[1]):
                factors = []
                config = {}
                # No fanout to inside mem for the innermost mem
                if org_idx == 0:
                    # if mem_idx < self.mapspace.shape[0] - 1:
                    if mem_idx > 0:
                        if self.arch.storage[mem_idx]['instances'] == self.arch.storage[mem_idx - 1]['instances']:
                            continue
                    elif mem_idx == 0:
                        continue

                config['target'] = self.arch.mem_name[mem_idx]
                config['type'] = self.org_name_dict[org_idx]
                for prob_idx in range(self.mapspace.shape[2]):
                    factors.append(self.mapspace[mem_idx][org_idx][prob_idx][1])
                    if self.mapspace[mem_idx][org_idx][prob_idx][1] != 1:
                        map_prob_bound[prob_idx] *= self.mapspace[mem_idx][org_idx][prob_idx][1]
                config['factors'] = self.generate_tile_str(factors)
                template_dict['mapspace']['constraints'].append(config)

        # Make sure the prob size for the mapping matches the orig prob size
        try:
            assert (map_prob_bound == self.prob.prob_bound[0:len(map_prob_bound)])
        except:
            logger.error("{} not equals to {}".format(map_prob_bound, self.prob.prob_bound[0:len(map_prob_bound)]))

    def generate_mapping(self):
        mapping = {}
        mapping['mapping'] = []
        map_prob_bound = [1] * self.mapspace.shape[2]

        for mem_idx in range(self.mapspace.shape[0]):
            for org_idx in range(self.mapspace.shape[1]):
                factors = []
                permutations = []
                config = {}
                # No fanout to inside mem for the innermost mem
                if org_idx == 0:
                    # if mem_idx < self.mapspace.shape[0] - 1:
                    if mem_idx > 0:
                        # if self.arch.storage[mem_idx]['instances'] == self.arch.storage[mem_idx-1]['instances']:
                        if self.arch.mem_instances[mem_idx] == self.arch.mem_instances[mem_idx - 1]:
                            continue
                    elif mem_idx == 0:
                        continue

                config['target'] = self.arch.mem_name[mem_idx]
                config['type'] = self.org_name_dict[org_idx]
                for prob_idx in range(self.mapspace.shape[2]):
                    permutations.append(self.mapspace[mem_idx][org_idx][prob_idx][0])
                    factors.append(self.mapspace[mem_idx][org_idx][prob_idx][1])
                    if self.mapspace[mem_idx][org_idx][prob_idx][1] != 1:
                        map_prob_bound[prob_idx] *= self.mapspace[mem_idx][org_idx][prob_idx][1]
                config['factors'] = self.generate_tile_str(factors)
                config['permutation'] = self.generate_perm_str(permutations)
                mapping['mapping'].append(config)

        # Make sure the prob size for the mapping matches the orig prob size
        try:
            assert (map_prob_bound == self.prob.prob_bound[0:len(map_prob_bound)])
        except:
            logger.error("{} not equals to {}".format(map_prob_bound, self.prob.prob_bound[0:len(map_prob_bound)]))

        for mem_idx in range(self.bypass.shape[0]):
            bypass = []
            keep = []
            config = {}
            config['target'] = self.arch.mem_name[mem_idx]
            config['type'] = 'bypass'
            for var_idx in range(self.bypass.shape[1]):
                if self.bypass[mem_idx][var_idx]:
                    bypass.append(self.var_name_dict[var_idx])
                else:
                    keep.append(self.var_name_dict[var_idx])
            config['bypass'] = bypass
            config['keep'] = keep
            mapping['mapping'].append(config)

        return mapping

    def print(self):
        print(self.__dict__)


class Mapping(object):
    """ Hardware Mapping"""

    def __init__(self):
        temporal = []
        spatial = []
        bypass = []


def gen_all_config(run, pickle_path=None, status_file=None, run_gen_map=True, run_gen_tc=True, run_sim_test=True):
    prob_path = pathlib.Path('timeloop_configs/01_3x3Conv/prob/toy_prob.yaml').resolve()
    prob = Prob(prob_path)

    arch_path = pathlib.Path('timeloop_configs/arch/simba.yaml').resolve()
    arch = Arch(arch_path)

    mapspace_path = pathlib.Path('timeloop_configs/00_1x1Conv/mapspace/mapspace.yaml').resolve()
    mapspace = Mapspace(mapspace_path)

    # init
    mapspace.init(prob, arch)

    # try to iterate from valid mapping
    # for prob_idx, mem_levels in enumerate(mapspace.factor_space):
    # for mem_level in mem_levels: # specify the mem_level of the current factor
    def is_config_done(factor_config, factor_space):
        prob_idx = len(factor_space) - 1
        mem_level = len(factor_space[prob_idx]) - 1
        if factor_config[prob_idx][mem_level] < factor_space[prob_idx][mem_level]:
            return False
        else:
            return True

    def config_increment(factor_config, factor_space, factor_org_idx):
        loop_carry = True
        for i, idx_tup in enumerate(factor_org_idx):
            prob_idx, mem_level = idx_tup
            # specify the mem_level of the current factor 
            if loop_carry:
                new_config = factor_config[prob_idx][mem_level] + 1
            if new_config < factor_space[prob_idx][mem_level]:
                loop_carry = False
                factor_config[prob_idx][mem_level] = new_config
                break
            else:
                if ((prob_idx == len(mapspace.factor_space) - 1) and
                        (mem_level == len(mem_levels) - 1)):
                    factor_config[prob_idx][mem_level] = new_config
                else:
                    factor_config[prob_idx][mem_level] = 0

    def get_factor_org_idx(factor_space):
        factor_org_idx = []
        for prob_idx, mem_levels in enumerate(factor_space):
            for mem_level, _ in enumerate(mem_levels):
                factor_org_idx.append((prob_idx, mem_level))
        return factor_org_idx

    # Compile simulation framework
    if run:
        x_size = 4
        y_size = 4
        utils.gen_makefile(x_size, y_size)
        utils.compile_sim()

    # key: config_space_str
    # val: [idx, run_gen_map, run_gen_tc, run_sim_cycle]
    # -1 means didnt run, positive value indicates different info
    # is_valid_mapping is determined with is_valid_mapping and timeloop run
    # sim_cycle: return by sim_test 
    if status_file:
        status_file = pathlib.Path(status_file)
        if status_file.is_file():
            status_dict = utils.parse_json(status_file)
    else:
        status_dict = {}

    configs = []
    if pickle_path:
        logger.info(pickle_path)

    perm_config = mapspace.get_default_perm()
    for spatial_config in mapspace.spatial_space:
        mapspace.reset_mapspace(spatial_config)
        factor_config = mapspace.get_default_factor()
        for prob_idx, mem_levels in enumerate(mapspace.factor_space):
            for mem_level, _ in enumerate(mem_levels):  # specify the mem_level of the current factor
                factor_config[prob_idx][mem_level] = 0
        logger.info("Init Iter: {} in {}".format(factor_config, mapspace.factor_space))

        factor_org_idx = get_factor_org_idx(mapspace.factor_space)
        idx = 0
        while (not is_config_done(factor_config, mapspace.factor_space)):

            if not mapspace.is_valid_mapping(factor_config, mapspace.factor_space):
                config_increment(factor_config, mapspace.factor_space, factor_org_idx)
                continue
            if run:
                logger.info("Iter: {} in {}".format(factor_config, mapspace.factor_space))
                if status_file:
                    run_config.run_config(mapspace, spatial_config, perm_config, factor_config, status_dict,
                                          run_gen_map, run_gen_tc, run_sim_test)
                else:
                    run_config.run_config(mapspace, spatial_config, perm_config, factor_config, dict(), run_gen_map,
                                          run_gen_tc, run_sim_test)
            # if pickle_path is defined, store the pickle config
            if pickle_path:
                configs.append((copy.deepcopy(spatial_config), copy.deepcopy(perm_config), copy.deepcopy(factor_config),
                                run_gen_map, run_gen_tc, run_sim_test))
            config_increment(factor_config, mapspace.factor_space, factor_org_idx)
            utils.delete_dramsim_log()
            idx += 1
    if status_file:
        utils.store_json(status_file, status_dict)

    if pickle_path:
        utils.store_pickle(pickle_path, configs)


if __name__ == "__main__":
    module_name = pathlib.Path(__file__).stem
    utils.setup_logging(module_name, logger)

    # unit test
    # assert(test_run())
    config_pkl = pathlib.Path('config.pkl')
    if not config_pkl.is_file():
        gen_all_config(False, pickle_path=config_pkl)
    configs = utils.parse_pickle(config_pkl)
