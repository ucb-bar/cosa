# NoC Simulator 

This a transaction-based NoC simulation we developed leveraging the Timeloop and Matchlib infrastcture to demonstrate the performance of the CoSA mapper.  
## 1. Environment Setup
Install Timeloop and Matchlib following the instructions in their repo. 

In matchlib, checkout the `hybrid_router` branch, copy the [sim/noc](https://github.com/ucb-bar/cosa/tree/main/src/nocsim/sim/noc) from this repo to [matchlib/cmod/unittests](https://github.com/NVlabs/matchlib/tree/hybrid_router/cmod/unittests). 

## 2. Run the Example 
### 2.1 Model the Workload in Timeloop
To generate the input for NoC, we need to first run Timeloop.
```
timeloop-model simba.yaml <mapping.yaml> <prob.yaml>
```
A `timeloop-model.map+stats.xml` file will be generated as in [this example](https://github.com/ucb-bar/cosa/blob/main/src/nocsim/runs/example/1_1_56_56_64_64_1/timeloop-model.map%2Bstats.xml)

### 2.2 Generate the Transactions
To run the simulation, run:
```
python gen_tc_io.py timeloop-model.map+stats.xml
```
A `nb.csv` file will be generated and we can pass it as input to the simulator. 

### 2.2 Generate the Transactions
```
cd matchlib/cmod/unittests/noc 
make 
./sim_test nb.csv > out
```
To see how long the simulation takes, search for `FINISH @` in the `out` file.
A `nb.json` will be generated and can be viewed in the Chrome browser [chrome://tracing](chrome://tracing). 

## 3 NoC Modeling
### 3.1 NoC Model
This NoC simulator supports a 2D mesh network with configurable X and Y size defined as `kNumNoCRouters_X` and `kNumNoCRouters_Y` in [sim/noc/RouterSpec.h](https://github.com/ucb-bar/cosa/tree/main/src/nocsim/sim/noc/RouterSpec.h).

https://github.com/ucb-bar/cosa/blob/f26e0d6af07e284a27128abe02f350a145a0cdba/src/nocsim/sim/noc/RouterSpec.h#L59

We found that the max number of flits is 5 (including the header) for Multicast.
This is because Multicast uses Cut-through routing.
**Cut-through routing** forwards a flit when the receiver has the space for the whole packet.
There is no limit on the number of flits per packet for Unicast as it uses Wormhole routing.
**Wormhole routing** forwards a flit when the receiver has the space for one flit.

Tunable Test Variables:
```
#define NUM_SRC 4
#define NUM_FLIT_PER_PACKET 5
#define NUM_PACKET 200
#define NUM_FLIT NUM_FLIT_PER_PACKET * NUM_PACKET
```

### 3.2 NoC Port Indexing Convention
The indexing of the nodes follows the convention below:
Assume  `kNumNoCRouters_X = 2` and `kNumNoCRouters_Y = 2`. The indices of ports are assigned as follows:
```
   4 5
11 0 1 6
10 2 3 7
   9 8
```

### 3.3 Transaction Format  
We define the events to simulate as transactions in the csv format.

| tc_id | actor_id | op | size | src | dest | dep |
|-------|----------|----|------|-----|------|-----|

Each transaction should have a unique `tc_id`.
`actor_id` specifies the srcdest port to perform the operation in the testbench.
Each transaction can only be launched afer all transactions specified in the `dep` list finish.
Note that the `src` should be the same as `actor_id` except it should be in the array format.

```
tc_id -- transaction id
actor_id -- current actor id
op -- operation: 0 for unicast, 1 for multicast, 2 for artificial cal with counter
size -- for op 0 and 1, the number of packets; for op 2, the number of cycle counts:
src -- list of src
dest -- list of dest
dep -- list of dependent tc_id
```

Example Transaction sequence: 
| tc_id | actor_id | op | size | src | dest | dep |
|-------|----------|----|------|-----|------|-----|
| 0     | 4        | 1  | 4    | 4   | 0,1  |     |
| 1     | 0        | 2  | 100  |     |      | 0   |
| 2     | 1        | 2  | 100  |     |      | 0   |
| 3     | 0        | 0  | 1    | 0   | 1    | 1   |

- tc 0 multicasts data of size 4 packets from port 4 to port 0 and 1. 
- tc 1 and 2 perform the calculation on port 0 and 1 for 100 cycles once the tc 0 finishes. 
- tc 3 unicasts the calculated data from port 0 to port 1 once tc 1 finishes. 


## 4. Transaction Generation 

### 4.1 PE and Traffic Extraction 
Given a Timeloop generated mapping shown below: 
```
DRAM [ Weights:147456 Inputs:115200 Outputs:100352 ]
----------------------------------------------------
| for P in [0:4)
|   for S in [0:3)
|     for C in [0:16) (Spatial-X)
InputBuffer [ Inputs:2016 ]
---------------------------
|       for N in [0:1)
|         for R in [0:3) (Spatial-X)
WeightBuffer [ Weights:1024 ]
-----------------------------
|           for Q in [0:28)
|             for P in [0:7)
AccumulationBuffer [ Outputs:128 ]
----------------------------------
|               for K in [0:128)
|                 for C in [0:8)
Registers [ Weights:1 ]
-----------------------
|                   for N in [0:1)

```

**Compute cycle** is product of inner PE temporal iterations. The PE compute cycle for the mapping above is `1x28x7x128x8x1 = 200704`. 
**Tensor transaction payload size** is the product of inner PE tensor related iterations. 
We assume double buffer optimization is implemented and aggregated all inner PE loops transactions for Weights and Outputs in one transaction to reduce the simulation costs. 

