#!/usr/bin/env python3 
import argparse
import logging
import os
import pathlib
import time
import re

import xml.etree.ElementTree as ET
import numpy as np

from utils import OrderedDefaultdict
import collections
import utils

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


def get_subnest_info(xml_file):
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

if __name__ == "__main__":
    output_base_path = pathlib.Path('output_dir_mapper').resolve()
    workload_base_path = pathlib.Path('prob').resolve() 
    workloads = ['conv', 'mm']
    arch_path = pathlib.Path('arch/arch.yaml').resolve()
    mapspace_path = pathlib.Path('mapspace/mapspace.yaml').resolve()

    for workload in workloads:
        workload_path = workload_base_path / workload
        unique_layers = utils.parse_yaml(workload_path / 'unique_layers.yaml')
        for unique_layer in unique_layers:
            layer_path = workload_path / (unique_layer+'.yaml')
            layer_path = layer_path.resolve()
            output_path = output_base_path / unique_layer
            utils.mkdir_p(output_path)
            os.chdir(output_path)
            print(layer_path)
            xml_file = output_path / 'timeloop-mapper.map+stats.xml'
            if not xml_file.exists(): 
                utils.run_timeloop_mapper(arch_path, layer_path, mapspace_path, cwd=output_path)
            
            if xml_file.exists(): 
                subnest_info = get_subnest_info(xml_file)
                cycle = subnest_info['cycle']
                energy = subnest_info['energy']
                print(f"Results cycle - {cycle}, energy - {energy}") 
            
            # break

