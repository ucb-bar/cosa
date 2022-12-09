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

#ifndef __HYBRIDROUTERTOP_H__
#define __HYBRIDROUTERTOP_H__

#include <systemc.h>
#include <nvhls_connections.h>
#include <nvhls_packet.h>
#include "RouterSpec.h"
#include <HybridRouter.h>

SC_MODULE(HybridRouterTop) {
 public:
  sc_in_clk clk; 
  sc_in<bool> rst;

  static const int NUM_LPORTS = spec::noc::numLPorts;
  static const int NUM_RPORTS = spec::noc::numRPorts;
  static const int NUM_VCHANNELS = spec::noc::numVChannels;
  static const int BUFFERSIZE = spec::noc::bufferSize;
  static const int MAX_HOPS = spec::noc::maxHops;
  static const int packetIDWidth = spec::noc::packetIDWidth;
  static const int flitDataWidth = spec::noc::flitDataWidth;

  static const int NoC2Dests = spec::kNumNoC2Routers;
  static const int NoCDests = spec::kNumNoCDests;
  static const int NoCMCastDests = spec::kNumNoCRouters;

  enum {
    kNumNoCRouters_X = spec::kNumNoCRouters_X,
    kNumNoCRouters_Y = spec::kNumNoCRouters_Y,
    kNumNoCMeshLPorts = spec::noc::kNumNoCMeshLPorts, 
    kNumNoCMeshRPorts = spec::noc::kNumNoCMeshRPorts, 
    kNumNoCMeshPorts = spec::noc::kNumNoCMeshPorts,

    NumPorts = NUM_LPORTS + NUM_RPORTS,
    NumCredits = (NUM_LPORTS + NUM_RPORTS) * NUM_VCHANNELS
  };
  typedef Flit<flitDataWidth, 0, 0, packetIDWidth, FlitId2bit, WormHole> Flit_t;
  typedef HybridRouter<NUM_LPORTS, NUM_RPORTS, BUFFERSIZE, Flit_t, NoC2Dests,
                       kNumNoCMeshPorts, NoCMCastDests, spec::noc::maxPktSize> Router;

  typedef Router::NoCRouteLUT_t NoCRouteLUT_t;
  typedef Router::NoC2RouteLUT_t NoC2RouteLUT_t;
  typedef Router::noc2_id_t noc2_id_t;
  typedef spec::noc::Credit_t Credit_t;
  typedef spec::noc::Credit_ret_t Credit_ret_t;

  static const int NOC_DEST_WIDTH = Router::NoC_dest_width;
  static const int NOC2_DEST_WIDTH = Router::NoC2_dest_width;
  static const int NOC_UCAST_DEST_WIDTH = Router::NoC_ucast_dest_width;
  static const int NOC2_UCAST_DEST_WIDTH = Router::NoC2_ucast_dest_width;
  static const int NOC_MCAST_DEST_WIDTH = Router::NoC_mcast_dest_width;
  static const int NOC2_MCAST_DEST_WIDTH = Router::NoC2_mcast_dest_width;
  static const int DEST_WIDTH = NOC_MCAST_DEST_WIDTH + NOC2_MCAST_DEST_WIDTH;

  // First declare routers, then pass the sc_module_name "routers" argument to the constructor
  // with HybridRouterTop constructor
  nvhls::nv_array<Router, kNumNoCRouters_X * kNumNoCRouters_Y> routers;
  Connections::In<Flit_t> in_port[kNumNoCMeshPorts];
  Connections::Out<Flit_t> out_port[kNumNoCMeshPorts];

  // it really should be Connections::Buffer
  // TODO figure out what to pass template<class Message, unsigned int NumEntries, Conections::connections_port_t port_marshall_type>
  Connections::Combinational<Flit_t> in_W_out_E[kNumNoCRouters_Y][kNumNoCRouters_X - 1];
  Connections::Combinational<Flit_t> in_E_out_W[kNumNoCRouters_Y][kNumNoCRouters_X - 1];
  Connections::Combinational<Flit_t> in_N_out_S[kNumNoCRouters_Y - 1][kNumNoCRouters_X];
  Connections::Combinational<Flit_t> in_S_out_N[kNumNoCRouters_Y - 1][kNumNoCRouters_X];

  // Return the ref to a router with index i, j 
  Router& router(size_t i, size_t j){
       return routers[i * kNumNoCRouters_X + j];
  }

  void connect_router_ports(size_t i, size_t j, size_t cur_router_port_id, size_t io_port_id) {
        router(i,j).in_port[cur_router_port_id](in_port[io_port_id]);
        router(i,j).out_port[cur_router_port_id](out_port[io_port_id]);
  }

  // Jtag

  // NoCRouteLUT_t is the number of bits 
  // needed for one-hot encoding of current router ports
  sc_in<NoCRouteLUT_t> NoCRouteLUT_jtags[spec::kNumNoCRouters][NOC_DEST_WIDTH]; // 0-15 
  sc_in<NoC2RouteLUT_t> NoC2RouteLUT_jtags[spec::kNumNoCRouters][NOC2_DEST_WIDTH]; // 16-19 
  // Set the id of current router
  //typedef 
  sc_in<NVUINTC(spec::kNoC2DestWidth)> noc2_id_jtags[spec::kNumNoCRouters]; // 20-23

  SC_HAS_PROCESS(HybridRouterTop);
  HybridRouterTop(sc_module_name name)
      : sc_module(name), clk("clk"), rst("rst"), routers("routers") {
      //: sc_module(name), clk("clk"), rst("rst"), router("router") {

    //cout << "spec::kNumNoCRouters*spec::kNumNoC2Routers " << spec::kNumNoCRouters*spec::kNumNoC2Routers << endl;
    //cout << "NOC_DEST_WIDTH " << NOC_DEST_WIDTH << endl;
    //cout << "NOC2_DEST_WIDTH " << NOC2_DEST_WIDTH << endl;

    // Connect clk and rst
    for (size_t i = 0; i < kNumNoCRouters_Y; i ++) {
        for (size_t j = 0; j < kNumNoCRouters_X; j ++) { 
            router(i, j).clk(clk); // port 6
            router(i, j).rst(rst); // port 7
        }
    }

    // Connect all local ports 
    for (size_t i = 0; i < kNumNoCRouters_Y; i ++) {
        for (size_t j = 0; j < kNumNoCRouters_X; j ++) { 
            size_t local_port_id = 0;
            size_t router_port_id = i * kNumNoCRouters_X + j;
            router(i, j).in_port[local_port_id](in_port[router_port_id]);
            router(i, j).out_port[local_port_id](out_port[router_port_id]);
        }
    }

    // Connect all horizontal interconnect
    for (size_t i = 0; i < kNumNoCRouters_Y; i ++) {
        for (size_t j = 0; j < kNumNoCRouters_X - 1; j ++) { 
            router(i, j).in_port[2](in_E_out_W[i][j]);
            router(i, j).out_port[2](in_W_out_E[i][j]);
            router(i, j+1).in_port[4](in_W_out_E[i][j]);
            router(i, j+1).out_port[4](in_E_out_W[i][j]);
        }
    } 

    // Connect all vertical interconnect
    for (size_t i = 0; i < kNumNoCRouters_Y - 1; i ++) {
        for (size_t j = 0; j < kNumNoCRouters_X; j ++) { 
            router(i, j).in_port[3](in_S_out_N[i][j]);
            router(i, j).out_port[3](in_N_out_S[i][j]);
            router(i+1, j).in_port[1](in_N_out_S[i][j]);
            router(i+1, j).out_port[1](in_S_out_N[i][j]);
        }
    } 

    size_t router_port_id = kNumNoCRouters_Y * kNumNoCRouters_X; 
    
    // Connect all mesh peripheral ports clock-wise
    for (size_t j = 0; j < kNumNoCRouters_X; j ++) { 
        connect_router_ports(0, j, 1, router_port_id);
        router_port_id ++;
    }
    for (size_t i = 0; i < kNumNoCRouters_Y; i ++) {
        connect_router_ports(i, kNumNoCRouters_X-1, 2, router_port_id);
        router_port_id ++;
    }
    for (size_t j = 0; j < kNumNoCRouters_X; j ++) { 
        connect_router_ports(kNumNoCRouters_Y-1, kNumNoCRouters_X-1-j, 3, router_port_id);
        router_port_id ++;
    }
    for (size_t i = 0; i < kNumNoCRouters_Y; i ++) {
        connect_router_ports(kNumNoCRouters_Y-1-i, 0, 4, router_port_id);
        router_port_id ++;
    }
  
    // NoC  
    for (int i = 0; i < kNumNoCRouters_Y; i ++) {
        for (int j = 0; j < kNumNoCRouters_X; j ++) {  
            int router_port_id = i * kNumNoCRouters_X + j;
            //cout << "Connect Router (" << i << "," << j << ") to jtag " << router_port_id << endl; 
            for (int k = 0; k < NOC_DEST_WIDTH; k++){
                //cout << "Connect NoC dest " << k << endl;
                router(i, j).NoCRouteLUT_jtag[k](NoCRouteLUT_jtags[router_port_id][k]); // Port 1-4
            }
        }
    }

    // NoP
    for (int i = 0; i < kNumNoCRouters_Y; i ++) {
        for (int j = 0; j < kNumNoCRouters_X; j ++) {  
            int router_port_id = i * kNumNoCRouters_X + j;
            //cout << "Connect Router (" << i << "," << j << ") to jtag " << router_port_id << endl; 
            for (int k = 0; k < NOC2_DEST_WIDTH; k++) {
                //cout << "Connect NoC2 dest " << k << endl;
                router(i, j).NoC2RouteLUT_jtag[k](NoC2RouteLUT_jtags[router_port_id][k]); // Port 5
            }
        }
    }

    // noc2_id_jtags
    for (int i = 0; i < kNumNoCRouters_Y; i ++) {
        for (int j = 0; j < kNumNoCRouters_X; j ++) {  
            int router_port_id = i * kNumNoCRouters_X + j;
            router(i, j).noc2_id_jtag(noc2_id_jtags[router_port_id]); // Port 0
        }
    }
  }

  // Delete the allocated routers
  ~HybridRouterTop() {}
};

#endif
