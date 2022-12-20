#!/usr/bin/env python3 
import argparse
import logging
import math
import collections 
import pathlib
import os 
import utils
from utils import dict_append_val
from parse_workload import *

rootLogger = logging.getLogger()
# rootLogger.setLevel(logging.DEBUG) # capture everything
# rootLogger.disabled = True


def construct_argparser():
    parser = argparse.ArgumentParser(description='Run Configuration')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        help='Output file path',
                        default='output',
                        )
    parser.add_argument('-i',
                        '--input_xml',
                        type=str,
                        help='input_xml',
                        default='timeloop-model.map+stats.xml',
                        )
    parser.add_argument('-p',
                        '--packet_size',
                        type=int,
                        help='packet size in bits',
                        default=256,
                        )

    return parser


# Let each TC only specify the Input and Output var and taking 
class Variable(object):
    def __init__(self, var_name, it, req_type, srcs, dests):
        self.var_name = var_name 
        # [mem_i, i]
        self.it = it
        self.req_type = req_type 
        self.srcs = srcs
        self.dests = dests

class Struct(object):
    def __init__(self, var_name, datawidth, entries):
        self.var_name = var_name
        self.datawidth = datawidth
        self.entries = entries


_struct_dict = {}


class TC(object): 
    """ Definition of each transaction"""
    def __init__(self, tc_id, actor_id, op, deps, tensor_name, entries, annotation=""):
        self.tc_id = tc_id
        self.actor_id = actor_id
        self.op = op
        self.size = 0 # number of packets
        self.deps = deps
        self.srcs = [actor_id]
        self.dests = []
        self.annotation = annotation
        self.tensor_name = tensor_name
        self.entries = 0
        self.datawidth = 0
        self.datastruct = None
        
        if op != COUNT:
            self.entries = entries # number of data items 
            self.datawidth = var_bits[tensor_name]
            struct_dict_key = (tensor_name, self.datawidth, entries) 
            if struct_dict_key in _struct_dict.keys():
                self.datastruct = _struct_dict[struct_dict_key] 
            else:
                new_struct = Struct(tensor_name, self.datawidth, entries)
                _struct_dict[struct_dict_key] = new_struct
                self.datastruct = new_struct

    def create_unicast(self, dest, size):
        self.create_multicast([dest], size)

    def create_multicast(self, dests, size):
        self.dests = dests
        self.size = size

    def create_count(self, size):
        self.size = size 

    def format_str(self, var): 
        var_str = ' '.join(str(i) for i in var)
        return var_str

    def format_csv(self): 
        assert(isinstance(self.deps, list))
        assert(isinstance(self.srcs, list))
        assert(isinstance(self.dests, list))
        # add comment 
        csv_str = "# {}\n".format(self.annotation) 

        # | tc_id | actor_id | op | size | src | dest | dep |
        csv_str += "{},{},{},{},{},{},{}".format(self.tc_id, 
                self.actor_id, self.op, self.size, 
                self.format_str(self.srcs), 
                self.format_str(self.dests), 
                self.format_str(self.deps) 
                )
        return csv_str
         

class TCEncoder(JSONEncoder):
        def default(self, o):
            return o.__dict__


class NoC(object): 
    """ Definition of each NoC"""
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_routers = X * Y
        self.num_lports = self.num_routers 
        self.num_rports = 4 * X * Y - Y * (X - 1) * 2 - X * (Y - 1) * 2 
        self.num_ports = self.num_lports + self.num_rports

        self.lports = list(range(self.num_lports)) 
        self.rports = list(range(self.num_lports, self.num_ports)) 

        fakeY = Y + 1 
        GB_PORT = os.getenv('GB_PORT')
        if GB_PORT is not None:
            self.globalbuf_port = GB_PORT
        else:
            self.globalbuf_port = X * fakeY - 1

        DRAM_PORT = os.getenv('DRAM_PORT')
        if DRAM_PORT is not None:
            self.dram_port = DRAM_PORT
        else:
            self.dram_port = X * fakeY + 4 * X * fakeY - fakeY * (X - 1) * 2 - X * (fakeY - 1) * 2 - (X + fakeY) 
        #self.dram_port = X * fakeY + 4 * X * fakeY - fakeY * (X - 1) * 2 - X * (fakeY - 1) * 2 - 1 
        #self.dram_port = X * fakeY - 2 

        rootLogger.info("Initialize NoC with X={0}, Y={1}, num_router={2}, num_lports={3}, num_rports={4}, globalbuf_port={5}, dram_port={6}".
                format(X, Y, self.num_routers, self.lports, self.rports, self.globalbuf_port, self.dram_port))


flit_size = 64
num_flit = 4 # total 5 but 1 for header
_PACKET_SIZE =  flit_size * num_flit
UNICAST, MULTICAST, COUNT = range(3)

class TC_Generator(object):   
    """ Definition of TC_Generator"""
    def __init__(self, num_spatial_cores):
        #X, Y = self.partition_cores(num_spatial_cores)
        X, Y = 4, 4
        self.noc = NoC(X, Y)
        self.tc_id = 0 
        self.tcs = []
        self.write_deps = {} 
        self.num_spatial_cores = num_spatial_cores 
        self.unicast_hops = 0
        self.multicast_hops = 0

    def partition_cores(self, num_spatial_cores):
        X = 1
        Y = num_spatial_cores
        for i in range(1, int(math.ceil(math.sqrt(num_spatial_cores)) + 1)):
            if num_spatial_cores % i == 0:
               X = i
               Y = num_spatial_cores // i
        return (Y, X)

    def get_xy_corrdinate(self, pe_id):
        fakeY = self.noc.Y + 1 
        if pe_id < self.noc.X * fakeY:
            x = pe_id % self.noc.X
            y = pe_id // self.noc.X
        elif pe_id < self.noc.X * fakeY + self.noc.X: 
            y = -1
            x = pe_id - self.noc.X * fakeY 
        elif pe_id < self.noc.X * fakeY + self.noc.X + fakeY:
            x = self.noc.X 
            y = pe_id - (self.noc.X * fakeY + self.noc.X)
        elif pe_id < self.noc.X * fakeY + fakeY + 2 * self.noc.X:
            y = fakeY
            x = (self.noc.X-1)-(pe_id-(self.noc.X * fakeY + self.noc.X + fakeY))
        elif pe_id < self.noc.X * fakeY + 2 * fakeY + 2 * self.noc.X:
            x = -1
            y = (fakeY-1)-(pe_id-(self.noc.X * fakeY + fakeY + 2 * self.noc.X))
        else:
            raise
        return (x, y)

    def count_hops_single(self, src, dest):
        src_x, src_y = self.get_xy_corrdinate(src)
        dest_x, dest_y = self.get_xy_corrdinate(dest)
        
        paths = []
        x_step = 1 if dest_x > src_x else -1
        for i in range(src_x, dest_x, x_step):
            src_xy = (i, src_y)
            dest_xy = (i+x_step, src_y)
            assert(i+x_step>=0 and i+x_step<self.noc.X)
            paths.append((src_xy, dest_xy))

        y_step = 1 if dest_y > src_y else -1
        for i in range(src_y, dest_y, y_step):
            src_xy = (dest_x, i)
            dest_xy = (dest_x, i+y_step)
            assert(i+y_step>=0 and i+y_step<self.noc.Y+2)
            paths.append((src_xy, dest_xy))

        return paths
        
    def count_hops(self, src, dests):
        path_set = set()
        #sep_hops = 0
        for dest in dests: 
            paths = self.count_hops_single(src, dest)
            #sep_hops += len(paths)
            path_set.update(paths)
        #print(path_set)
        #print(len(path_set))
        #print(sep_hops)
        return len(path_set)

    def unicast(self, tensor_name, size, src, dest, deps, var): 
        """ 
        Send data from src to dest
            tensor_name - name of the tensor not variable with id 
            size - data size in bit = datawidth * # of entries
            deps - list of depedent tc ids 
            var - output var to produce by this func
        """
        dep_tcs = []

        num_packets = (size - 1) // _PACKET_SIZE + 1
        entries = size // var_bits[tensor_name]
        annotation = "{}: unicast from {} to {} dep on {}, {} bits in {} packets".format(var, src, dest, deps, size, num_packets)
        tc = TC(self.tc_id, src, UNICAST, deps, tensor_name, entries, annotation)
        tc.create_unicast(dest, num_packets)
        dep_tcs.append(self.tc_id)
        self.tcs.append(tc)

        self.write_deps[var] = dep_tcs
        if src == self.noc.globalbuf_port: 
            hops = self.count_hops(src, [dest])
            self.unicast_hops += hops + num_packets * _PACKET_SIZE // flit_size
        #if dest == self.noc.globalbuf_port: 
        #    hops = self.count_hops(src, [dest])
        #    self.unicast_hops += hops + num_packets * _PACKET_SIZE // flit_size

        self.tc_id += 1

    def multicast(self, tensor_name, size, src, dests, deps, var):
        """ 
        Send data from src to dests
            tensor_name - name of the tensor not variable with id 
            size - data size in bit 
            deps - list of depedent tc ids 
            var - output var to produce by this func
        """
        dep_tcs = []

        num_packets = (size - 1) // _PACKET_SIZE + 1
        entries = size // var_bits[tensor_name]
        # TODO assume data is of multiple packet size
        # rem_size = size 
        # while(rem_size > 0):
        #     if rem_size > _PACKET_SIZE:
        #         send_size = _PACKET_SIZE 
        #     else:
        #         send_size = rem_size
        #     rem_size = rem_size - send_size
        annotation = "{}: multicast from {} to {} dep on {}, {} bits in {} packets".format(var, src, dests, deps, size, num_packets)
        tc = TC(self.tc_id, src, MULTICAST, deps, tensor_name, entries, annotation)
        tc.create_multicast(dests, num_packets)
        dep_tcs.append(self.tc_id)
        self.tcs.append(tc)
        self.write_deps[var] = dep_tcs
        if src == self.noc.globalbuf_port: 
            #hops = self.count_hops(src, dests)
            all_hops = [self.count_hops(src, [dest]) for dest in dests]
            hops = max(all_hops) 
            #self.multicast_hops += hops * num_packets
            self.multicast_hops += hops + num_packets * _PACKET_SIZE // flit_size
        self.tc_id += 1
    
    def gather(self, tensor_name, size, srcs, dest, deps, var):
        """ 
        Gather data from srcs to dest
            tensor_name - name of the tensor not variable with id 
            size - data size in bit 
            deps - list of depedent tc ids 
            var - output var to produce by this func
        """
        dep_tcs = []
        entries = size // var_bits[tensor_name]
        for i, src in enumerate(srcs): 
            self.unicast(tensor_name, size, src, dest, deps, var+'_'+str(i))
            deps = self.get_deps([var+'_'+str(i)])
            dep_tcs.extend(deps)
        self.write_deps[var] = dep_tcs

    def count(self, size, src, deps, var): 
        """ 
        Run PE counters on src
            size - number of cycles
            deps - list of depedent tc ids 
            var - output var to produce by this func
        """
        dep_tcs = []

        annotation = "{}: {} count {} cycles dep on {}".format(var, src, size, deps)
        tc = TC(self.tc_id, src, COUNT, deps, "", 0, annotation)
        tc.create_count(size)
        dep_tcs.append(self.tc_id)
        self.tcs.append(tc)
        self.tc_id += 1

        self.write_deps[var] = dep_tcs

    # Get all var reqs names  
    def get_reqs(self, var, dep_reqs, pe_id):
        # a bit of hack here, if var has __, it needs to be matched exactly
        # otherwise match the prefix only 
        if "__" in var:
            assert(var in self.write_deps.keys())
            dep_reqs.append(var) 
        else: 
            for k in self.write_deps.keys():
                if k.split("__")[0] == var:
                    #rootLogger.info("var {}".format(var))
                    #rootLogger.info("pe_id {}".format(pe_id))
                    if pe_id == -1: 
                        dep_reqs.append(k)
                    else:
                        ids_str = k.split("_")[-1]
                        ids = map(int, ids_str.split("-"))
                        if pe_id in ids:
                            dep_reqs.append(k)

    # Return the TC the current req is dependent on  
    def get_deps(self, dep_vars, pe_id=-1):
        deps = []
        dep_reqs = []
        
        # get the matched dep reqs, 
        # 1. if var has __, it needs to be matched exactly
        # 2. otherwise match the prefix
        #   if pe_id is not -1, also match the PE id that is related to the req
        #   else match all req with the prefix 
        assert (type(dep_vars) is list)
        for var in dep_vars:
            self.get_reqs(var, dep_reqs, pe_id)
            # rootLogger.info("var {}".format(var))
            # rootLogger.info("write_deps {}".format(self.write_deps))
            # rootLogger.info("dep_reqs {}".format(dep_reqs))

        # get transaction id from write_deps
        for var in dep_reqs:
            deps.extend(self.write_deps[var])
        return deps
    
    def to_file(self, out_file): 
        if out_file.suffix == '.csv':
            csv_str = "\n".join( tc.format_csv() for tc in self.tcs) 
            with open(out_file, 'w') as f:
                f.write(csv_str)
        elif out_file.suffix == '.json':
            with open(out_file, "w") as f:
                json.dump(self.tcs, f, indent=" ", cls=TCEncoder)
        else:
            raise("Not supported output file format!")
             

# Construct unicast for PEs accessing different addr 
# Construct broadcast for PEs accessing the same addr 
# pe_dep_vars specifies the dep_vars that the dst matches the current pe
# dep_vars specifies the dep_vars that just var name matches 
def construct_send_reqs(var_name, tc, buf, tensor_name, data_size, pe_dep_vars, dep_vars=[], mem_port=None):

    if mem_port is None:
        mem_port = tc.noc.globalbuf_port
    
    addrs = construct_addrs_dict(buf)
    rootLogger.debug("construct_send_reqs: addrs {}".format(addrs))

    # i is just iter idx
    for i, (k, v) in enumerate(addrs.items()):
        if len(v) == 1: 
            pe = v[0] 
            deps = [] 
            deps.extend(tc.get_deps(pe_dep_vars, pe)) 
            deps.extend(tc.get_deps(dep_vars)) 

            output_var = var_name+"__req_send_unicast_"+str(pe)
            tc.unicast(tensor_name, data_size, mem_port, pe, deps, output_var)
            rootLogger.info("\t{0} from Global Memory Port {1} to PE {2}".format(output_var, mem_port, v[0]))
        else: 
            pes = v
            pes_str = '-'.join(map(str, pes))
            deps = [] 
            for pe in pes:
                deps.extend(tc.get_deps(pe_dep_vars, pe)) 
            deps.extend(tc.get_deps(dep_vars)) 

            output_var = var_name+"__req_send_multicast_"+str(pes_str)
            tc.multicast(tensor_name, data_size, mem_port, pes, deps, output_var)              
            rootLogger.info("\t{0} Global Memory Port {1} to PE {2}".format(output_var, mem_port, v))


# X, Y is the dimension of the NoC
def xy_reduction(addrs, X, Y):
    pe_deps = {}
    pe_sends = {}
    ret_pes = []
    for i, (k, v) in enumerate(addrs.items()):
        # if output to the same addr, need to be reduced
        row_arr = [[] for i in range(Y)] 
        red_arr = []

        # stores pes on the same row to row_arr
        for j, pe in enumerate(v):
            row_idx = pe // X
            row_arr[row_idx].append(pe)

        # reduce across the row dim
        for j, row in enumerate(row_arr):
            last_pe = None
            num_pes = len(row) 
            for k, pe in enumerate(row):
                if k != 0:
                    dict_append_val(pe_deps, pe, last_pe)
                    dict_append_val(pe_sends, last_pe, pe)
                last_pe = pe
                if k == num_pes - 1:
                    red_arr.append(pe)
        
        last_pe = None
        num_red_pes = len(red_arr) 
        # reduce across the red_arr
        for j, pe in enumerate(red_arr):
            if j != 0:
                dict_append_val(pe_deps, pe, last_pe)
                dict_append_val(pe_sends, last_pe, pe)
            last_pe = pe
            if j == num_red_pes - 1:
                ret_pes.append(pe)

    rootLogger.debug("pe_deps: {}, pe_sends: {}, ret_pes: {}".format(pe_deps, pe_sends, ret_pes))
    return (pe_deps, pe_sends, ret_pes)
                    
def serial_reduction(addrs):
    # Vars: 
    # - pe_deps is a dict storing pe and the pes it deps on
    # - pe_sends is the reverse dictionary 
    # - ret_pes is the reduction node that stores the results
    # Returns: (pe_deps, pe_sends, ret_pes) 
    
    pe_deps = {}
    pe_sends = {}
    ret_pes = []
    for i, (k, v) in enumerate(addrs.items()):
        # if output to the same addr, need to be reduced
        num_pes = len(v)
        if num_pes > 1: 
            last_pe = None
            for j, pe in enumerate(v):
                if j != 0:
                    pe_deps[pe] = [last_pe]
                    pe_sends[last_pe] = [pe]
                last_pe = pe
                if j == num_pes - 1:
                    ret_pes.append(pe)
    return (pe_deps, pe_sends, ret_pes)
                    

# Construct unicast for partial sum reduction 
def construct_reduce_reqs(var_name, tc, buf, tensor_name, data_size, dep_vars, reduction="xy"):

    # construct addrs dict
    addrs = construct_addrs_dict(buf) 
    rootLogger.debug("construct_reduce_reqs: addrs {}".format(addrs))

    # construct reduction pattern 
    if reduction == "serial":
        pe_deps, pe_sends, ret_pes = serial_reduction(addrs)
    elif reduction == "xy": 
        pe_deps, pe_sends, ret_pes = xy_reduction(addrs, tc.noc.X, tc.noc.Y)
    else:
        rootLogger.critical("{} is not valid reduction".format(reduction))
        raise

    for src_pe, dst_pes in pe_sends.items(): 
        deps = []
        deps.extend(tc.get_deps(dep_vars, src_pe))
        if src_pe in pe_deps.keys():
            # Find the dependencies for the src_pe 
            for dep_pe in pe_deps[src_pe]:
                dep_var = "{0}__req_reduce_unicast_{1}_{2}".format(var_name, dep_pe, src_pe) 
                new_deps = tc.get_deps([dep_var])
                deps.extend(new_deps)
            
        # assume always one dst for reduction
        assert(len(dst_pes)==1)
        for dst_pe in dst_pes: 
            output_var = "{0}__req_reduce_unicast_{1}_{2}".format(var_name, src_pe, dst_pe) 
            tc.unicast(tensor_name, data_size, src_pe, dst_pe, deps, output_var)
            rootLogger.info("{}".format(output_var))
            rootLogger.info("\t{0} from PE {1} to PE {2}".format(output_var, src_pe, dst_pe))

    return ret_pes
#     for i, (k, v) in enumerate(addrs.items()):
#         # if output to the same addr, need to be reduced
#         num_pes = len(v)
#         if num_pes > 1: 
#             for j, pe in enumerate(v):
#                 deps = []
#                 for var in dep_vars:
#                     dep.extend(tc.get_deps(var, pe))
#                 # add deps if not the first pe
#                 if j != 0: 
#                     dep_vars = [var_name]
#                     red_deps = tc.get_deps(dep_vars)
#                     red_deps.extend(deps)
# 
#                 # if not the last pe, send to the next pe
#                 if j != num_pes - 1:
#                     tc.unicast(data_size, pe, v[j+1], red_deps, var_name+"__req_reduce_unicast" + str(pe))
#                     #rootLogger.info("{}".format(tc.write_deps))
#                     rootLogger.info("{}".format(var_name+"__req_reduce_"+str(j)+"_unicast"))
#                     rootLogger.info("\treq_reduce_{0}_unicast from PE {1} to PE {2}".format(i, pe, v[j+1]))


def construct_todram_reqs(var_name, tc, ret_pes, tensor_name, data_size, dep_vars):
    deps = tc.get_deps(dep_vars)
    dest = tc.noc.dram_port

    # send to GlobalBuffer from the ret PEs
    for i, pe in enumerate(ret_pes):
        output_var = var_name+"__req_store_unicast_"+str(pe)
        tc.unicast(tensor_name, data_size, pe, dest, deps, output_var)
        rootLogger.info("\t{0} from PE {1} to DRAM Port {2}".format(output_var, pe, dest))
    

def construct_toglobalbuf_reqs(var_name, tc, ret_pes, tensor_name, data_size, dep_vars):
    deps = tc.get_deps(dep_vars)
    dest = tc.noc.globalbuf_port

    # send to GlobalBuffer from the ret PEs
    for i, pe in enumerate(ret_pes):
        output_var = var_name+"__req_store_unicast_"+str(pe)
        tc.unicast(tensor_name, data_size, pe, dest, deps, output_var)
        rootLogger.info("\t{0} from PE {1} to GlobalBuffer Port {2}".format(output_var, pe, dest))


def combine_schedule(mem_schedule, noc_schedule, weight_schedule, out_file = 'tc.csv'):
    deps = []

    rootLogger.info("================ Generate TCs to {0} ================".format(out_file))

    tc = TC_Generator(noc_schedule['num_spatial_cores'])

    mem_temporal_loops = mem_schedule['temporal_loops']
    mem_temporal_loop_iter = mem_schedule['temporal_loop_iter']

    noc_temporal_loops = noc_schedule['temporal_loops']
    noc_temporal_loop_iter = noc_schedule['temporal_loop_iter']

    last_output_store = None

    print_step_i = 0 
    step_i = 0
    for mem_i in range (mem_schedule['num_steps']):
        loop_indices_str = ', '.join(str(idx) for idx in mem_temporal_loop_iter)
        rootLogger.info("---------------- Mem Temporal Step {0} ({1}) ----------------".format(mem_i, loop_indices_str))
    
        # A. Send from DRAM to GlobalBuffer
        for var_name in ['Outputs', 'Inputs']: 
            data_size = mem_schedule['data_size'][var_name] * var_bits[var_name]
            if mem_schedule['steps'][var_name][mem_i] != 0:
                rootLogger.info("Send {0} from DRAM to GlobalBuffer".format(var_name))
                dep_vars = []

                # TODO Optimization Enforce write after read to PE buffers? double buffering here
                # the dependency should be when i-2 output send finish
                if mem_i > 1:
                    dep_vars.append("Outputs_Update_{}_{}".format(mem_i-2, noc_schedule['num_steps'] -1))

                # construct broadcast and point-to-point req based the addrs
                var_name_annotate = "MEM{}_{}".format(var_name, mem_i)
                output_var = var_name_annotate+"__req_send_unicast_"+str(tc.noc.globalbuf_port)
                deps = tc.get_deps(dep_vars)
                tc.unicast(var_name, data_size, tc.noc.dram_port, tc.noc.globalbuf_port, deps, output_var)

        
        base_i = mem_i * noc_schedule['num_steps']
        for i in range(noc_schedule['num_steps']):
            step_i += 1
            if step_i // 100 != print_step_i:
                print_step_i = step_i // 100
                print(step_i)

            loop_indices_str = ', '.join(str(idx) for idx in noc_temporal_loop_iter)
            rootLogger.info("---------------- NoC Temporal Step {0} ({1}) ----------------".format(i, loop_indices_str))
            increment(noc_temporal_loops, noc_temporal_loop_iter)

            # 1. Send from GlobalBuffer to PE
            for var_name in ['Outputs', 'Inputs']: 
                if noc_schedule['steps'][var_name][i] != 0:
                    rootLogger.info("Send {0} from GlobalBuffer to PEs".format(var_name))
                    dep_vars = []

                    data_size = noc_schedule['data_size'][var_name] * var_bits[var_name]
                    #if var_name == "Weights" or var_name == "Inputs" or var_name == "Outputs":
                        # TODO IMPORTANT ! If C is in the temperal DRAM level, need to load partial sum or not? 
                        # Assume K spatial partition, K in the outer temperal loop,
                        # and C in the outer temperal loop, need to reload partial sum 

                    # TODO Optimization Enforce write after read to PE buffers? double buffering here
                    # the dependency should be when i-2 output send finish
                    if i > 1:
                        dep_vars.append("Outputs_Update_{}_{}".format(mem_i, i-2))

                    # TODO Refactor this code!! 
                    gb_dep_vars = [] 
                    for prev_mem_i in range(mem_i+1):
                        gb_dep_vars.append("MEM{}_{}".format(var_name, prev_mem_i))
                    # TODO This is too strict
                    # Enforce write after writer order to PE buffers, send 
                    #    if i > 0: 
                    #        dep_vars.append(var_name + "_" + str(i-1))
                            
                    # construct broadcast and point-to-point req based the addrs
                    construct_send_reqs("{}_{}_{}".format(var_name, mem_i, i), tc, noc_schedule['buf_spatial'][var_name], var_name, data_size, dep_vars, gb_dep_vars, tc.noc.globalbuf_port)
            
            # 1. Check if there is need to load partial sum
            var_name = "Weights"
            if weight_schedule['steps'][var_name][base_i + i] != 0:
                rootLogger.info("Send {0} from DRAM to PEs".format(var_name))
                dep_vars = []
                if i > 1:
                    dep_vars.append("Outputs_Update_{}_{}".format(mem_i, i-2))

                data_size = weight_schedule['data_size'][var_name] * var_bits[var_name]
                construct_send_reqs("{}_{}_{}".format(var_name, mem_i, i), tc, weight_schedule['buf_spatial'][var_name], var_name, data_size, dep_vars, mem_port=tc.noc.dram_port)


            # 2. PE execution
            # Enforce read after write order and start counting 
            dep_vars = [ "{}_{}_{}".format(dep_var, mem_i, i) for dep_var in var_dep["Outputs_Update"]] 
            
            # Assume double buffer here
            if i > 1: 
                dep_vars.append("Outputs_Reduce_{}_{}".format(mem_i, i-2))
                dep_vars.append("Outputs_Store_{}_{}".format(mem_i, i-2))
                prev_dep_vars =  [ "{}_{}_{}".format(dep_var, mem_i, i-2) for dep_var in var_dep["Outputs_Update"]]
                dep_vars.extend(prev_dep_vars)
            elif i == 1:
                prev_dep_vars =  [ "{}_{}_{}".format(dep_var, mem_i, i-1) for dep_var in var_dep["Outputs_Update"]]
                dep_vars.extend(prev_dep_vars)


            rootLogger.info("Count {0} Cycles".format(noc_schedule['pe_cycle']))
            # Add non blocking threads for PEs  
            for src_pe in tc.noc.lports:
                # add individual count and dep check 
                deps = tc.get_deps(dep_vars, src_pe)
                pe_var = "Outputs_Update_{}_{}__req_count_{}".format(mem_i, i, str(src_pe))
                tc.count(noc_schedule['pe_cycle'], src_pe, deps, pe_var) 

            # 3. Write back to GlobalBuffer
            # data_size in bytes
            data_size = noc_schedule['data_size']['Outputs']
                
            var_name = "Outputs_Store"
            #if noc_schedule['steps'][var_name][i] != 0:
            if noc_schedule['steps'][var_name][i] != 0:
                rootLogger.info("Store Outputs to GlobalBuffer")
                # Construct Reduce for C partitioning
                # TODO make the reduction logic more configurable
                # for CCC temporal organization, we can skip the reduction logic
                rootLogger.info("Spatial Reduction for Partial Sum".format(noc_schedule['pe_cycle']))
                dep_vars = ["Outputs_Update_{}_{}".format(mem_i, i)]
                # in-network reduction width is 24 bits
                tensor_name = "Outputs"
                ret_pes = construct_reduce_reqs("Outputs_Reduce_{}_{}".format(mem_i, i), tc, noc_schedule['buf_spatial']['Outputs'], tensor_name, data_size * var_bits['Outputs'], dep_vars)

                dep_vars = ["Outputs_Update_{}_{}".format(mem_i, i), "Outputs_Reduce_{}_{}".format(mem_i, i)]  
                if i > 0:
                    dep_vars.append("{}_{}_{}".format(var_name, mem_i, i-1))
                # if C is not completed, we cannot quantize, assume 24 bits for now 
                # if assume quantized, output bits should be 8 bits
                tensor_name = "Outputs"
                construct_toglobalbuf_reqs("{}_{}_{}".format(var_name,mem_i, i), tc, ret_pes, tensor_name, data_size * var_bits['Outputs'], dep_vars)
                last_output_store = "{}_{}_{}".format(var_name,mem_i, i)
            
        # C. Store outputs from GlobalBuffer to DRAM
        var_name = "Outputs_Store"
        data_size = mem_schedule['data_size']['Outputs'] * var_bits[var_name]
        if mem_schedule['steps'][var_name][mem_i] != 0:
            rootLogger.info("Send {0} from GlobalBuffer to DRAM".format(var_name))
            dep_vars = []
            # TODO Optimization Enforce write after read to PE buffers? double buffering here
            # the dependency should be when i-2 output send finish
            if last_output_store is not None:
                dep_vars.append(last_output_store)
                last_output_store = None

            # construct broadcast and point-to-point req based the addrs
            # construct_send_reqs("MEM{}_{}".format(var_name, mem_i), tc, mem_schedule['buf_spatial'][var_name], data_size, dep_vars)
            var_name = "MEM{}_{}".format(var_name, mem_i)
            output_var = var_name+"__req_send_unicast_"+str(tc.noc.dram_port)
            deps = tc.get_deps(dep_vars)
            tensor_name = "Outputs"
            tc.unicast(tensor_name, data_size, tc.noc.globalbuf_port, tc.noc.dram_port, deps, output_var)

    tc.to_file(out_file)
    return tc

def test_tc_hop_count():
    tc = TC_Generator(16)
    total_hops = tc.count_hops(19, list(range(4)))
    print(total_hops)

def test_tc():
    out_file = 'tc.csv'

    tc = TC_Generator(16)

    # Send data form GlobalBuffer to all cores 
    tc.dram_to_all(2**8, [], "Inputs")
    tc.dram_to_all(2**8, [], "Weights")

    # Get dep tc_id for  Inputs and Weights
    dep_vars = ["Inputs", "Weights"]
    deps = tc.get_deps(dep_vars)


    # Perform PE counter on all cores 
    tc.all_count(2**10, deps, "Outputs")

    # Get dep tc_id for Outputs 
    # and send data to GlobalBuffer
    deps = tc.get_deps(["Outputs"])
    tc.all_to_dram(2**8, deps, "Output_GlobalBuffer")
    
    tc.to_file(out_file)


def gen_tc(xml_file, out_file, outer_loopcount_limit = None):
    subnest_info = get_subnest_info(xml_file);
    
    timeout = False
    outer_loopcount = get_outer_temp_loopcount(subnest_info, start_level=4) 
    print("outer_loopcount: {}".format(outer_loopcount))
    if outer_loopcount_limit is not None:
        if outer_loopcount > outer_loopcount_limit:
            timeout = True
            return (timeout, None)

    mem_schedule = gen_schedule(subnest_info, start_level=5) 
    noc_schedule = gen_schedule(subnest_info, start_level=4, end_level=5)
    weight_schedule = gen_schedule(subnest_info, start_level=4)
    cost = copy.deepcopy(noc_schedule['cost'])

    dram_latency = 17
    # Weight traffic 
    cost['Weights'] = weight_schedule['cost']['Weights'] * dram_latency
    cost['Weights_milp'] = weight_schedule['cost']['Weights_milp'] # * dram_latency
    for var in ['Inputs', 'Outputs', 'Outputs_Store']:
        cost[var] += mem_schedule['cost'][var] * dram_latency

    cost['Total'] = 0
    for var in ['Weights', 'Inputs', 'Outputs', 'Outputs_Store']:
        cost['Total'] += cost[var]

    tc = combine_schedule(mem_schedule, noc_schedule, weight_schedule, out_file)
    return (timeout, (cost, (tc.unicast_hops, tc.multicast_hops)))


if __name__ == "__main__":
    # module_name = os.path.basename(__file__).replace(".py", "")
    # utils.setup_logging(module_name, rootLogger)  

    parser = construct_argparser()
    args = parser.parse_args()


    xml_file = pathlib.Path(args.input_xml) 
    out_file = pathlib.Path(args.output)
    _PACKET_SIZE = args.packet_size
    
    gen_tc(xml_file, out_file)
