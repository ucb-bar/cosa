/*
 * Copyright (c) 2016-2019, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __ROUTER_SPEC__
#define __ROUTER_SPEC__

#include <nvhls_int.h>
#include <nvhls_types.h>

namespace spec {

//#if defined(NOC_X) && defined(NOC_Y)
const int kNumNoCRouters_X = NOC_X;
const int kNumNoCRouters_Y = NOC_Y;
//#else 
//const int kNumNoCRouters_X = 8;
//const int kNumNoCRouters_Y = 8;
//#endif

// NoC level routers, mcast max 20
const int kNumNoCRouters = kNumNoCRouters_X * kNumNoCRouters_Y;
const int kNumNoCDests = kNumNoCRouters;

// Assume 2nd-level NoP is one, mcast max 36  
const int kNumNoC2Routers = 1;

// Decimal-encoding first level 
const int kNoCDestWidth = nvhls::index_width<kNumNoCDests>::val;  
// Decimal-encoding second level 
const int kNoC2DestWidth = nvhls::index_width<kNumNoC2Routers>::val;  

namespace noc {

static const int flitDataWidth = 64;
static const int flitTCWidth = 24; // we have 21 bits for cast dest, 43-bits for packet: 19 bit for packet and 24 bit for TC ID 
static const int packetIDWidth = 0;
static const int flitIDWidth = 2;
static const int numVChannels = 1;
static const int numLPorts = 1;
static const int numRPorts = 4;
static const int maxHops = 16;
static const int maxPktSize = 5;
static const int bufferSize = maxPktSize + 2;

// Local Input Ports for the NoC mesh 
const int kNumNoCMeshLPorts = kNumNoCRouters_X * kNumNoCRouters_Y; 

// Remote Input Ports for the NoC mesh, assume numRPorts is 4 
const int kNumNoCMeshRPorts = 4 * kNumNoCRouters_X * kNumNoCRouters_Y\
                                - kNumNoCRouters_Y * (kNumNoCRouters_X - 1) * 2\
                                - kNumNoCRouters_X * (kNumNoCRouters_Y - 1) * 2;

const int kNumNoCMeshPorts = kNumNoCMeshLPorts + kNumNoCMeshRPorts;
static const int log_bufferSize = nvhls::index_width<bufferSize + 1>::val;
typedef NVUINTC(log_bufferSize) Credit_t;
typedef NVUINTC(1) Credit_ret_t;

}  // namespace noc
}  // namespace spec

#endif
