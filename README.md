# CoSA: Scheduling by Constrained Optimization for Spatial Accelerators
CoSA is a scheduler for spatial DNN accelerators that generate high-performance schedules in one shot using mixed integer programming (MIP).
For more details, please refer to:
- [ISCA'21 CoSA Paper](https://arxiv.org/pdf/2105.01898.pdf)
- [ISCA'21 CoSA Presentation](https://people.eecs.berkeley.edu/~qijing.huang/2021ISCA/2021ISCA_CoSA_Presentation.pdf)

CoSA leverages the regularities in DNN operators and hardware to formulate the DNN scheduling space into a MIP problem with algorithmic and architectural constraints, which can be solved to automatically generate a highly efficient schedule in one shot.

## Installation
0. CoSA 
Download the CoSA source code `
git clone git@github.com:ucb-bar/cosa.git
`
1. Gurobi
Please follow the instructions in [Gurobi for Academics and Researchers](https://www.gurobi.com/academia/academic-program-and-licenses/) to install Gurobi optimizer and obtain an academic license.
2. Timeloop
Please refer to the instructions in the [Timeloop Tutorial](http://accelergy.mit.edu/infra_instructions.html) to install Timeloop with Docker.
To install from source code please, follow the instructions in [Timeloop Github](https://github.com/NVlabs/timeloop).
The specific Timeloop version used for CoSA evaluation is commit [11920be](https://github.com/NVlabs/timeloop/commit/11920be5a744239c985ff049256f2fc40f65ce8b).
3. Python3
Install packages for Python3 with:
```
pip install numpy==1.19.0 PyYAML==5.3.1 yamlordereddictloader==0.4.0 seaborn==0.10.1 matplotlib==3.2.2 pandas==1.0.5
```
4. Environment Set##up 
Update `env.sh` with the paths to `COSA_DIR`, `TIMELOOP_DIR`, `GUROBI_HOME`, and `GRB_LICENSE_FILE`, and source the environment `source env.sh`.

## Run CoSA

To run one scheduling example:
```
python3 src/cosa.py
```

## CoSA Inputs and Outputs
CoSA takes problem dimension, architecture constraints, relation encoding constants as inputs and returns 
a mapping with tiling, temporal/spatial, and permutation solved to optimize the user defined objective.  
```
def cosa(prob, arch, A, B, part_ratios, global_buf_idx, Z=None): 
    """Run CoSA to generate a mapping with tiling, temporal/spatial, and permutation determined. 
        We currently assume there is a global buffer 
    Args:
        prob: An object defines the layer dimension.
        arch: An object defines the hardware architecture dimension. 
        A: A 2d binary constant matrix that encodes the layer dimension to data tensor relationship.
            1 means related, 0 means unrelated
            Note that the R,S to the input tensor relation is specially handled in the formulation,
            and are specified to 0. 
        B: A 2d binary constant matrix that encodes the data tensor to memory level mapping. 
            It can be derived from the mapspace bypass pattern in Timeloop. 
            Note it is intended to be used for even mapping among different data tensors to different memory levels.
        part_ratios: A 2d array to represent the partition ratios of different data tensors in different memory buffers. 
        global_buf_idx: An index point to the global buffer. 
        Z: Similar to B, but intended for uneven mapping among different data tensors to different memory levels.
            It is a 3d binary constant matrix that encodes the data tensor to memory level mapping.

    Returns: 
        factor_config: A 2d array specifying the allocation decision for each prime factor.
        spatial_config: A 2d array specifying the temporal/spatial decisions for each prime factor.
        perm_config: A 2d array specifying the ordering of R,S,P,Q,C,K,N factors at each 
    """
```

## Even and Uneven Mapping
CoSA shall be able to support the even (using matrix B to encode bypassing scheme in [Timeloop](https://github.com/NVlabs/timeloop)) and uneven mapping (using matrix Z to encode rank to memory mapping for different data tensors as in [ZigZag](https://github.com/ZigZag-Project/zigzag)) 
