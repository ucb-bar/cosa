import yaml
import pathlib
import utils
import copy

mm_configs=[(1024, 1024, 1024),
        (512, 32, 512),
        (512, 512, 512),
        (128, 128, 128),
        (128, 512, 128),
        (128, 2048, 512),
        (128, 512, 2048),
        (128, 512, 512)
        ]

conv_configs=[
        (7,7,224,224,3,64,1,2,2),
        (1,1,56,56,64,64,1,1,1),
        (3,3,14,14,256,256,1,1,1),
        (3,3,7,7,512,512,1,1,1),
        ]



d = {'R':1,'S':1,'P':1,'Q':1,'C':1,'K':1,'N':1,'Wstride':1,'Hstride':1,'Wdilation':1,'Hdilation':1}

conv_key = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'N', 7:'Wstride', 8:'Hstride'}
mm_key = {0:'P', 1:'K', 2:'C'}

def gen_config(d, configs, op_name, op_key):
    config_dir=pathlib.Path(f'prob/{op_name}')
    utils.mkdir_p(config_dir)
    unique_layers = []
    for idx, item in enumerate(configs):
        prob_fn = f'{op_name}_{idx}'
        unique_layers.append(prob_fn) 
        new_d = copy.deepcopy(d)
        for i, v in enumerate(item):
            key = op_key[i]
            new_d[key] = v
            path = config_dir / (prob_fn+'.yaml')
            prob_d = {}
            prob_d['problem'] = new_d
            utils.store_yaml(path, prob_d)
        print(new_d)
    path = config_dir / ('unique_layers.yaml')
    utils.store_yaml(path, unique_layers)


gen_config(d, conv_configs, 'conv', conv_key)
gen_config(d, mm_configs, 'mm', mm_key)
