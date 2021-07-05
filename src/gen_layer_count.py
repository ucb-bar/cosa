import utils
import pathlib
from cosa_input_objs import Prob, Arch, Mapspace

def is_unique_prob(prob, probs):
    for exist_prob in probs:
        if prob == exist_prob:
            return False
    return True

def get_unique_layers():
    model_strs = ['alexnet', 'resnet50', 'resnext50_32x4d', 'deepbench']
    workload_dir = pathlib.Path('../configs/workloads').resolve()
    
    for model_str in model_strs:
        model_dir = workload_dir / (model_str+'_graph')

        layer_def_path = model_dir / 'layers.yaml'
        layers = utils.parse_yaml(layer_def_path)

        # layer_dicts = []
        # unique_layers= []
        layer_count = {}
        for layer in layers:
            prob_path = model_dir / (layer + '.yaml') 
            prob = Prob(prob_path)
            config_str = prob.config_str()
            layer_count[config_str] = layer_count.get(config_str, 0) + 1
            
            # if is_unique_prob(prob.prob, layer_dicts): 
            #     layer_dicts.append(prob.prob)
            #     unique_layers.append(layer)

        # layer_def_path = model_dir / 'unique_layers.yaml'
        layer_def_path = model_dir / 'layer_count.yaml'
        utils.store_yaml(layer_def_path, layer_count)

        total = sum(layer_count.values())
        print(model_str + f" layers counted : {total}")

get_unique_layers()
