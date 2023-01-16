import torch
from torchvision import models 
import torchsummary 

import pathlib
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

import os
import yaml, errno


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def store_yaml(yaml_path, data):
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)


def parse_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.full_load(f)
        return data


def compose_prob(file_path, prob_dict):
    input_dict = {}
    input_dict["problem"] = prob_dict
    input_dict['problem']['shape'] = 'cnn-layer'
    store_yaml(file_path, input_dict)


def get_unique_layers(model_strs, workload_dir):
    
    d = {}
    for model_str in model_strs:
        model_dir = workload_dir / (model_str+'_graph')

        layer_def_path = model_dir / 'layers.yaml'
        layers = parse_yaml(layer_def_path)

        unique_layers= []
        for layer in layers:
            prob_path = model_dir / (layer + '.yaml') 
            prob = parse_yaml(prob_path)
            key = tuple(sorted(prob.items())) 
            if key in d.keys():
                d[key] +=1
            else:
                d[key] = 1
                unique_layers.append(layer)

        layer_def_path = model_dir / 'unique_layers.yaml'
        store_yaml(layer_def_path, unique_layers)
    for k,v in d.items():
        if v != 1:
            print(f'{k}: {v}')


## Adopt from torchsummary
def summary_string(model, input_size, batch_size=-1, device=torch.device('cpu:0'), dtypes=None):
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            # workaround for not derefed tensor
            if type(input[0]) == list:
                input = input[0]
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
                summary[m_key]['weight'] = list(module.weight.size())
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]['bias'] = list(module.weight.size())
            if hasattr(module, "stride"):
                summary[m_key]['stride'] = module.stride
            if hasattr(module, "dilation"):
                summary[m_key]['dilation'] = module.dilation

            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()
    return summary


def generate_prob_json(summary, output_dir):
    layer_list = []
    for layer in summary:
        print("======={}=======".format(layer))

        if 'Conv' in layer or 'Linear' in layer:
            yaml_path = output_dir / "{}.yaml".format(layer)
            layer_list.append(layer)
            print("weight: {}".format(summary[layer]['weight']))
            print("output_shape: {}". format(summary[layer]['output_shape']))

            prob_dict = {'R': None, 'S': None, 'P':None, 'Q':None, 'C':None, 'K':None, 'N':None, 'Wstride': 1, 'Hstride': 1, 'Wdilation': 1, 'Hdilation': 1}
            
            prob_dict['N'] = summary[layer]['output_shape'][0]
            if 'Conv' in layer:
                print("stride: {}".format(summary[layer]['stride']))
                print("dilation: {}".format(summary[layer]['dilation']))

                prob_dict['R'] = summary[layer]['weight'][2]
                prob_dict['S'] = summary[layer]['weight'][3]
                prob_dict['P'] = summary[layer]['output_shape'][2]
                prob_dict['Q'] = summary[layer]['output_shape'][3]
                prob_dict['C'] = summary[layer]['weight'][1]
                prob_dict['K'] = summary[layer]['weight'][0]
                prob_dict['Wstride'] = summary[layer]['stride'][0]
                prob_dict['Hstride'] = summary[layer]['stride'][1]
                prob_dict['Wdilation'] = summary[layer]['dilation'][0]
                prob_dict['Hdilation'] = summary[layer]['dilation'][1]
            else: 
                prob_dict['R'] = 1
                prob_dict['S'] = 1
                prob_dict['P'] = 1 
                prob_dict['Q'] = 1 
                prob_dict['C'] = summary[layer]['weight'][1]
                prob_dict['K'] = summary[layer]['weight'][0]

            store_yaml(yaml_path, prob_dict)
    layer_def_path = output_dir / 'layers.yaml'
    store_yaml(layer_def_path, layer_list)


def generate_prob(model_str, batch_size, output_dir):
    prob_dir = output_dir / f'{model_str}_graph'
    mkdir_p(prob_dir)
    call_model = getattr(models, model_str)
    model = call_model()
    input_size = (3,224, 224) 

    model_summary = summary_string(model, input_size, batch_size)
    #torchsummary.summary(model, input_size)
    generate_prob_json(model_summary, prob_dir)


if  __name__ == "__main__":

    model_strs = ['alexnet', 'resnet50', 'resnext50_32x4d', 'densenet161', 'vgg16']
    output_dir = pathlib.Path('workloads')
    
    for model in model_strs:
        generate_prob(model, 1, output_dir)
    get_unique_layers(model_strs, output_dir)

