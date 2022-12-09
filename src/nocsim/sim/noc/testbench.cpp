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

#include "Events.h"
#include "Device.h"

#include "HybridRouterTop.h"
#include "RouterSpec.h"
#include "TCTracker.h"
#include <systemc.h>
#include <mc_scverify.h>
#include <testbench/Pacer.h>
#include <testbench/nvhls_rand.h>
#include <nvhls_connections.h>

#define NVHLS_VERIFY_BLOCKS (HybridRouterTop)
#define NUM_SRC 1
#define NUM_FLIT_PER_PACKET 1
#define NUM_PACKET 1
#define NUM_FLIT NUM_FLIT_PER_PACKET * NUM_PACKET 

#include <nvhls_verify.h>

#include <map>
#include <vector>
#include <deque>
#include <utility>
#include <sstream>

using namespace ::std;
typedef HybridRouterTop::Flit_t Flit_t;
typedef HybridRouterTop::Credit_t Credit_t;
typedef HybridRouterTop::Credit_ret_t Credit_ret_t;
const unsigned int NUM_LPORTS = spec::noc::numLPorts;
const unsigned int NUM_RPORTS = spec::noc::numRPorts;
const unsigned int NUM_VCHANNELS = spec::noc::numVChannels;
const unsigned int BUFFERSIZE = spec::noc::bufferSize;
//const int NOC_DEST_WIDTH = HybridRouterTop::NOC_DEST_WIDTH;
const int NOC_DEST_WIDTH = HybridRouterTop::kNumNoCMeshPorts;
const int NOC2_DEST_WIDTH = HybridRouterTop::NOC2_DEST_WIDTH;
const int NOC_UCAST_DEST_WIDTH = HybridRouterTop::NOC_UCAST_DEST_WIDTH;
const int NOC2_UCAST_DEST_WIDTH = HybridRouterTop::NOC2_UCAST_DEST_WIDTH;
const int NOC_MCAST_DEST_WIDTH = HybridRouterTop::NOC_MCAST_DEST_WIDTH;
const int NOC2_MCAST_DEST_WIDTH = HybridRouterTop::NOC2_MCAST_DEST_WIDTH;
const int DEST_WIDTH = NOC_MCAST_DEST_WIDTH + NOC2_MCAST_DEST_WIDTH;

typedef deque<Flit_t> flits_t;

static const int kNumNoCMeshLPorts = HybridRouterTop::kNumNoCMeshLPorts;
static const int kNumNoCMeshRPorts = HybridRouterTop::kNumNoCMeshRPorts;
static const int kNumNoCMeshPorts = HybridRouterTop::kNumNoCMeshPorts;

std::string getEnvVar( std::string const & key )
{
        char * val = getenv( key.c_str() );
            return val == NULL ? std::string("") : std::string(val);
}

//#if defined(DRAM_PORT)
//static const int DRAMPort = kNumNoCMeshLPorts - 2;
//static const int DRAMPort = kNumNoCMeshPorts - 1;
//static const int DRAMPort = DRAM_PORT 
//#else 
static size_t DRAMPort = kNumNoCMeshPorts - (spec::kNumNoCRouters_X + spec::kNumNoCRouters_Y);
//#endif

static const int kNumNoCRouters_X = spec::kNumNoCRouters_X;
static const int kNumNoCRouters_Y = spec::kNumNoCRouters_Y;

// Wrong way to use shared var
//static int num_recv_flits = 0;

enum RouteType {
  UniCast,
  UniCast_NoCongestion,
  UniCast_SingleDest,
  MultiCast,
  MultiCast_SingleSource,
  MultiCast_DualSource
};

// Return the router port for peripheral src
int get_fixed_port(int id, bool is_onehot){
  int router_port;
  if (id < kNumNoCMeshLPorts)
      router_port = is_onehot? 1 : 0;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X)
      router_port = is_onehot? 8 : 3;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X + kNumNoCRouters_Y)
      router_port = is_onehot? 16 : 4;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X*2 + kNumNoCRouters_Y) 
      router_port = is_onehot? 2 : 1;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X*2 + kNumNoCRouters_Y*2) 
      router_port = is_onehot? 4: 2;
  else {
      cout << "Invalid id " << id << endl;
      assert(false);
  }
  return router_port;
}


// Return the router port for peripheral dest
int get_router_port(int id, bool is_onehot){
  int router_port;
  if (id < kNumNoCMeshLPorts)
      router_port = is_onehot? 1 : 0;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X)
      router_port = is_onehot? 2 : 1;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X + kNumNoCRouters_Y)
      router_port = is_onehot? 4 : 2;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X*2 + kNumNoCRouters_Y) 
      router_port = is_onehot? 8 : 3;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X*2 + kNumNoCRouters_Y*2) 
      router_port = is_onehot? 16: 4;
  else {
      cout << "Invalid id " << id << endl;
      assert(false);
  }
  return router_port;
}


// Get the id for the corresponding router for the peripheral src/dest
int get_router_id(int id){
  assert(id < kNumNoCMeshLPorts + kNumNoCRouters_X*2 + kNumNoCRouters_Y*2);
  assert(kNumNoCMeshPorts == kNumNoCMeshLPorts + kNumNoCRouters_X*2 + kNumNoCRouters_Y*2);

  int router_id; 
  if (id < kNumNoCMeshLPorts)
      router_id = id;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X)
      router_id = id - kNumNoCMeshLPorts;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X + kNumNoCRouters_Y)
      router_id = (id - (kNumNoCMeshLPorts + kNumNoCRouters_X) + 1) * kNumNoCRouters_X - 1;
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X*2 + kNumNoCRouters_Y) { 
      int local_id = id - (kNumNoCMeshLPorts + kNumNoCRouters_X + kNumNoCRouters_Y);
      int reverse_id = kNumNoCRouters_X - 1 - local_id;
      router_id = reverse_id + (kNumNoCRouters_Y-1) * kNumNoCRouters_X;   
  }
  else if (id < kNumNoCMeshLPorts + kNumNoCRouters_X*2 + kNumNoCRouters_Y*2) { 
      int local_id = id - (kNumNoCMeshLPorts + kNumNoCRouters_X*2 + kNumNoCRouters_Y);
      int reverse_id = kNumNoCRouters_Y - 1 - local_id;
      router_id = reverse_id * kNumNoCRouters_X;   
  } else {
      cout << "Invalid id " << id << endl;
      assert(false);
  }
  return router_id;
}

void get_router_loc(int router_id, int& src_x, int& src_y){
  // router idx is organized as 0, ..., kNumNoCRouters_X-1, 
  //                            kNumNoCRouters_X, ..., 2*kNumNoCRouters_X-1
  src_x = router_id % kNumNoCRouters_X;
  src_y = router_id / kNumNoCRouters_X;
}


// Get the dest id given the router_id and output port
int get_dest_id(int router_id, int port){
    int id = 0;
    int src_x, src_y, dest_x, dest_y; 
    get_router_loc(router_id, src_x, src_y);

    if (port == 0) {
        id = router_id;
    }  else if (port == 1) { // North port dest 
      dest_x = src_x;
      dest_y = src_y - 1;
      if (dest_y < 0) {
          id = kNumNoCMeshLPorts + dest_x; 
      } else {
          id = dest_y * kNumNoCRouters_X  + dest_x;  
      }
    }  else if (port == 2) { // East port dest 
      dest_x = src_x + 1;
      dest_y = src_y;
      if (dest_x >= kNumNoCRouters_X) {
          id = kNumNoCMeshLPorts + kNumNoCRouters_X + dest_y; 
      } else {
          id = dest_y * kNumNoCRouters_X  + dest_x;  
      }
    }  else if (port == 3) { // South port dest 
      dest_x = src_x;
      dest_y = src_y + 1;
      if (dest_y >= kNumNoCRouters_Y) {
          id = kNumNoCMeshLPorts + kNumNoCRouters_X + kNumNoCRouters_Y + (kNumNoCRouters_X - 1 - dest_x); 
      } else {
          id = dest_y * kNumNoCRouters_X  + dest_x;  
      }
    }  else if (port == 4) { // West port dest 
      dest_x = src_x - 1;
      dest_y = src_y;
      if (dest_x < 0) {
          id = kNumNoCMeshLPorts + 2*kNumNoCRouters_X + 2*kNumNoCRouters_Y - 1 - dest_y;
      } else {
          id = dest_y * kNumNoCRouters_X  + dest_x;  
      }
    }
    return id;
}

int get_port(int src_id, int dest_id, bool is_onehot) {
      NVUINTW(DEST_WIDTH) route;
      int src_router_id = get_router_id(src_id);
      int dest_router_id = get_router_id(dest_id);
      int src_x, src_y, dest_x, dest_y; 
      get_router_loc(src_router_id, src_x, src_y);
      get_router_loc(dest_router_id, dest_x, dest_y);

      CDCOUT("ROUTE DEBUG"<< endl, kDebugLevel); 
      CDCOUT("src_id: " << src_id << ", dest_id: " << dest_id << endl, kDebugLevel); 
      CDCOUT("src_router_id: " << src_router_id << ", dest_router_id: " << dest_router_id << endl, kDebugLevel); 
      int x_diff = dest_x - src_x;
      int y_diff = dest_y - src_y;
      CDCOUT("x_diff: " << x_diff << ", y_diff: " << y_diff << endl, kDebugLevel); 

      int port; 

      //int L = is_onehot? 1 : 0;
      int N = is_onehot? 2 : 1;
      int E = is_onehot? 4 : 2;
      int S = is_onehot? 8 : 3;
      int W = is_onehot? 16: 4;

      if (x_diff > 0) {
          port = E;
          CDCOUT("router port E" << endl, kDebugLevel);
      } else if (x_diff < 0) {
          port = W;
          CDCOUT("router port W" << endl, kDebugLevel);
      } else { // If equal
          if (y_diff > 0) {
            port = S;
            CDCOUT("router port S" << endl, kDebugLevel);
          } else if (y_diff < 0) {
            port = N;
            CDCOUT("router port N" << endl, kDebugLevel);
          } else {
            // For dest at the boundary, port might not be 0 
            port = get_router_port(dest_id, is_onehot);
            if (port == 0) {
                CDCOUT("router port L" << endl, kDebugLevel);
            } else {
                CDCOUT("local port with " << port << " in onehot " << is_onehot << endl, kDebugLevel);
            }
          }
      }
      return port;
}

class Reference {
 public:
  Reference() { init_tracking(); }

  ~Reference() {
    //cout << "ref send_cnt = " << send_cnt << endl;
    //cout << "ref recv_cnt = " << recv_cnt << endl;
    printf("ref send_cnt = %d\n", send_cnt);
    printf("ref recv_cnt = %d\n", recv_cnt);
  }

  unsigned send_cnt, recv_cnt;
  void flit_sent(const int& id, const deque<int>& dest, const int& out_vc,
                 const Flit_t& flit) {
    for (deque<int>::const_iterator it = dest.begin(); it != dest.end(); ++it) {
      CDCOUT(hex << sc_time_stamp() << " sent flit: " << flit, kDebugLevel);
      CDCOUT(" added to ref_tracking[" << *it << "][" << out_vc << "][" << id
           << "]" << dec << endl, kDebugLevel)
      ref_tracking[*it][out_vc][id].push_back(flit);
      ++send_cnt;
    }
  }

  void flit_received(const int& id, const Flit_t& flit) {
    flit_received_testing(id, flit, dest_state[id]);
    ++recv_cnt;
  }

  void print_dest_ref(const int& id) {
    CDCOUT("--------------------" << endl, kDebugLevel);
    for (int inport = 0; inport < kNumNoCMeshPorts; ++inport) {
      for (unsigned vc = 0; vc < NUM_VCHANNELS; ++vc) {
        CDCOUT("ref_tracking[" << id << "][" << vc << " ][" << inport
             << "].top= ", kDebugLevel) ;

        if (ref_tracking[id][vc][inport].empty()) {
          CDCOUT("EMPTY" << endl, kDebugLevel);
        } else {
          for (deque<Flit_t>::iterator it =
                   ref_tracking[id][vc][inport].begin();
               it != ref_tracking[id][vc][inport].end(); ++it) {
            Flit_t flit = *it;  // ref_tracking[id][vc][inport].front();
            CDCOUT(flit << endl, kDebugLevel);
          }
        }
      }
    }
    CDCOUT("--------------------" << endl, kDebugLevel);
  }

 private:
  typedef vector<int> vc_chain_t;
  typedef vector<flits_t> vc_dst_storage_t;  // each entry is storage from a
                                             // given source - the whole
                                             // structure is for a single vc
  typedef vector<vc_dst_storage_t> dst_storage_t;  // storage per vc
  typedef vector<dst_storage_t>
      tracking_t;  // each entry is a destination being tracked

  tracking_t ref_tracking;

  void init_tracking() {
    //ref_tracking.resize(HybridRouterTop::NumPorts);
    ref_tracking.resize(kNumNoCMeshPorts);

    for (int dest = 0; dest < kNumNoCMeshPorts; ++dest) {
      ref_tracking[dest].resize(NUM_VCHANNELS);
      for (unsigned vc = 0; vc < NUM_VCHANNELS; ++vc) {
        ref_tracking[dest][vc].resize(kNumNoCMeshPorts);
      }
    }

    dest_state.resize(kNumNoCMeshPorts);

    send_cnt = recv_cnt = 0;
  }

  struct dest_state_t {
    dest_state_t() : source(NUM_VCHANNELS){};
    vector<deque<int> > source;  // list of potential sources for each VC
    Flit_t extracted_header[NUM_VCHANNELS];  // temporary storage for extracted
                                             // header
  };

  vector<dest_state_t> dest_state;

  void flit_received_testing(const int& id, const Flit_t& flit,
                             dest_state_t& ds) {
    int flit_packet_id = 0;
    flit_packet_id = flit.get_packet_id();
    if (flit.flit_id.isHeader()) {

      ds.source[flit_packet_id].clear();
      // find out who sent this flit - go over all sources and check
      for (int i = 0; i < kNumNoCMeshPorts; ++i) {
        if (!ref_tracking[id][flit_packet_id][i].empty()) {
          // there are pending flits to arrive from source i, did this flit
          // arrive from source i?
          // this check assumes unique match
          Flit_t inspected_flit = ref_tracking[id][flit_packet_id][i].front();
          int inspected_flit_packet_id = inspected_flit.get_packet_id();
          assert(inspected_flit_packet_id == flit_packet_id);
          if ((flit.flit_id == inspected_flit.flit_id) &&
              ((flit.data >> (DEST_WIDTH + 1)) ==
               (inspected_flit.data >> (DEST_WIDTH + 1)))) {
            ds.source[flit_packet_id].push_back(i);
            ds.extracted_header[flit_packet_id] = flit;

            // JENNY: only one sender matches, but what if two headers are received consecutively? 
            ref_tracking[id][flit_packet_id][i].pop_front();
            CDCOUT("DEBUG: flit.flit_id " << flit.flit_id << ", inspected_flit.flit_id " << inspected_flit.flit_id << endl, kDebugLevel); 
            CDCOUT("DEBUG: flit.data " << flit.data << ", inspected_flit.data " << inspected_flit.data << endl, kDebugLevel); 
            CDCOUT("DEBUG: compare data " << (flit.data >> (DEST_WIDTH + 1)) << ", inspected_data " << (inspected_flit.data >> (DEST_WIDTH + 1)) << endl, kDebugLevel); 
            CDCOUT("DEBUG: popping [" << id << "][" << flit_packet_id << "][" << i << "]" << endl, kDebugLevel);
          }
        }
      }
      assert(!ds.source[flit_packet_id].empty());
      if (flit.flit_id.isSingle()) {
        assert(ds.source[flit_packet_id].size() == 1);
        // there will be no body flits following, remove it from sources list
        // right now
        ds.source[flit_packet_id].clear();
      }
    } else {

      if (ds.source[flit_packet_id].size() > 1) {
        bool found = false;
        int found_source;
        // multiple headers did match, find which one is correct
        for (deque<int>::iterator it = ds.source[flit_packet_id].begin();
             it != ds.source[flit_packet_id].end(); ++it) {
          bool matching = false;
          if (!ref_tracking[id][flit_packet_id][*it].empty()) {
            Flit_t expected_flit =
                ref_tracking[id][flit_packet_id][*it].front();
            assert(flit_packet_id == expected_flit.get_packet_id());
            if ((flit.flit_id == expected_flit.flit_id) &&
                (flit.data == expected_flit.data)) {
              assert(found == false);
              found = true;
              matching = true;
              found_source = *it;
            }
          }

          if (!matching) {
            // put back the extracted flit
            // JENNY: reset the connection if not matching?
            ref_tracking[id][flit_packet_id][*it].push_front(
                ds.extracted_header[flit_packet_id]);
          }
        }

        assert(found);
        ds.source[flit_packet_id].clear();
        ds.source[flit_packet_id].push_back(found_source);
        // only a single candidate should remain at the very end
      }
      CDCOUT( hex << "Dest ID: : " << id
           << " Source ID: " << ds.source[flit_packet_id].front()
           << " Received: " << flit, kDebugLevel);
      assert(
          !ref_tracking[id][flit_packet_id][ds.source[flit_packet_id].front()]
               .empty());

      Flit_t expected_flit =
          ref_tracking[id][flit_packet_id][ds.source[flit_packet_id].front()]
              .front();

      CDCOUT( " Expected: " << expected_flit << dec << endl, kDebugLevel);
      assert(flit_packet_id == expected_flit.get_packet_id());
      assert(flit.flit_id == expected_flit.flit_id);
      assert(flit.data == expected_flit.data);

      ref_tracking[id][flit_packet_id][ds.source[flit_packet_id].front()]
          .pop_front();
    }
  }
};

SC_MODULE(SrcDest) {
  Connections::Out<Flit_t> out;
  Connections::In<Flit_t> in;
  sc_in<bool> clk;
  sc_in<bool> rst;
  const unsigned int id;
  Pacer pacer;
  Reference& ref;
  TCTracker& tc_tracker;
  s_ptr<EventManager> em_ptr;
  s_ptr<Device> device_ptr;
  s_ptr<DDR>ddr_ptr; 

  // dram read enable
  bool dram_ren=true;

  // dram write enable
  bool dram_wen=true;
   
  // indicate whether dram simulation is enabled
  bool dram_en() { return dram_ren || dram_wen;}

  int select_vc() {
    // randomally select a vc that has >0 credits
    // ok also not to select any some probablity

    deque<int> valid_vc;

    for (unsigned i = 0; i < NUM_VCHANNELS; i++) {
      valid_vc.push_back(i);
    }

    if (valid_vc.empty()
#ifndef DISABLE_PACER
        || ((rand() % 10) < 3)  // occasionally don't pick any vc
#endif
        )
      return -1;
    return valid_vc[rand() % valid_vc.size()];
  }

  void make_connection(s_ptr<Device> device_ptr, const std::string &mod_a, 
          const std::string &port_a, const std::string &mod_b, const std::string &port_b) {
      auto a = device_ptr->GetChild(mod_a);
      auto b = device_ptr->GetChild(mod_b);
      auto bus = device_ptr->GetChild(mod_a + '-' + mod_b);
      ConnectPorts(a, port_a, bus, "0");
      ConnectPorts(b, port_b, bus, "1");
  }

  void run() {
    // For DRAMSim
    PortId port_id; 
    Clock clk;

    // reset
    vector<flits_t> packet(NUM_VCHANNELS);
    deque<Address> dram_read_addrs;
    deque<Transaction*> tc_queue;
    // use for actually load data from DRAM, but not used for now
    //deque<Flit_t> dram_read_packet;
    deque<Flit_t> dram_write_packet;
    deque<Flit_t> dram_tail_flits;
    vector<deque<int> > dest_per_vc(NUM_VCHANNELS);

    deque<sc_core::sc_time> counter_start_time;
    int out_vc;
    in.Reset();
    out.Reset();
    int num_flit = 0;

    // DRAMSim Init
    if (id == DRAMPort && dram_en()) {
        em_ptr = std::make_shared<EventManager>();
        device_ptr = std::make_shared<Device>("nvdl", em_ptr); 

        s_ptr<SRam> llc_ptr = std::make_shared<SRam>("LLC", device_ptr, 1, 0, 0);
        device_ptr->AddChild("LLC", llc_ptr);
        
        s_ptr<CrossBar>cb_ptr = std::make_shared<CrossBar>("CrossBar", device_ptr, 1, 0);
        device_ptr->AddChild("CrossBar", cb_ptr);

        ddr_ptr = std::make_shared<DDR>("DDR", device_ptr, 1, 0, 0);
        device_ptr->AddChild("DDR", ddr_ptr);

        s_ptr<DMA>mc_ptr = std::make_shared<DMA>("MemCtrl", device_ptr, 0);
        device_ptr->AddChild("MemCtrl", mc_ptr);

        s_ptr<Bus> clbus_ptr = std::make_shared<Bus>("CrossBar-LLC", device_ptr, 0);
        device_ptr->AddChild("CrossBar-LLC", clbus_ptr);
       
        s_ptr<Bus> mlbus_ptr = std::make_shared<Bus>("MemCtrl-LLC", device_ptr, 0);
        device_ptr->AddChild("MemCtrl-LLC", clbus_ptr);

        s_ptr<Bus> mdbus_ptr = std::make_shared<Bus>("MemCtrl-DDR", device_ptr, 0);
        device_ptr->AddChild("MemCtrl-DDR", mdbus_ptr);

        make_connection(device_ptr, "CrossBar", "B0", "LLC", "0");
        make_connection(device_ptr, "CrossBar", "B0", "LLC", "0");
        make_connection(device_ptr, "MemCtrl", "0", "LLC", "0");
        make_connection(device_ptr, "MemCtrl", "1", "DDR", "0");
                                                                            
        device_ptr->start();
        s_ptr<Block> cbar = GetChildTyped<CrossBar>(device_ptr, "CrossBar");
        s_ptr<Block> ddr = device_ptr->GetChild("DDR");
        port_id = ddr->GetPortId("0");
        em_ptr->WatchPort(port_id);
        em_ptr->RegisterForClockTick(ddr);

//      Address addr = 0x90000;
//      for (int i=0; i < 1000; i++) {
//          if (i == 10) {
//            auto event = Event(i , port_id, IODirection::In, addr, 32, true);
//            em_ptr->QueueEvent(event);
//            cout << "TESTDRAM: Write to DRAM!!" << endl;
//          } else if (i == 15) {
//              Address addr = 0x30000;
//              auto event = Event(i, port_id, IODirection::In, addr, 16384 * 4, false);
//              em_ptr->QueueEvent(event);
//              cout << "TESTDRAM: Read from DRAM!!" << endl;
//          }
//          em_ptr->ClockTick(i);
//          auto e = em_ptr ->  GetWatchedEvent();  
//          while(e.is_valid) {
//                cout << "TESTDRAM: Received event clock:" << e.clock 
//                << " port_id:" << e.port_id << " address:"
//               << e.address << " is_write:" << e.is_write 
//               << " data_size: " << e.data_size << endl;
//            e = em_ptr ->  GetWatchedEvent(); 
//
//          }
//
//      }
    }


    Flit_t recv_flit;
    unsigned dram_recv_flits = 0;
 
    int print_tc_id = 0;
    while (1) {
      wait();
      clk = (sc_time_stamp().to_double() - 50) / 1000;
      CDCOUT( "===================" << id << "====================" << endl, kDebugLevel);
      if (in.PopNB(recv_flit)) {
        CDCOUT( "@" << sc_time_stamp() << hex << ": " << name()
             << " received flit: " << recv_flit << endl, kDebugLevel);
        ref.flit_received(id, recv_flit);

        if (id == DRAMPort && dram_wen) {
            dram_write_packet.push_back(recv_flit);
        } else {
            // flit_receive will delete the tc if finish recv packets 
            tc_tracker.flit_received(id, recv_flit, sc_time_stamp());
            //num_recv_flits += 1;
        }
      }

      // for DRAM write, only write when there are 32 bytes received 
      while (dram_wen && !dram_write_packet.empty()) { 
        Flit_t dram_flit = dram_write_packet.front(); 

        Transaction* tc_ptr = tc_tracker.get_flit_tc(dram_flit);
        if (dram_flit.flit_id.isHeader() && tc_ptr->get_dram_write_offset(id) == 0) {
            tc_ptr->dram_write_start_cycle_multi[id] = sc_time_stamp();
        }
        
        dram_recv_flits += 1;
        // keep track of the headers 
        if (dram_flit.flit_id.isTail()) {
            // assume for now all flits are aligned to 256 bits boundary for DRAM simulation
            assert(dram_recv_flits == spec::noc::maxPktSize);
            // packetSize in bytes for only data not including header flit
            unsigned packetSize = (dram_recv_flits - 1) * (spec::noc::flitDataWidth >> 3); 
            // transactionSize in bytes 
            assert(ddr_ptr->transactionSize == packetSize);
            //re
            Address addr = 0x00000;
            // for every DRAM receive, write back to DRAM 
            auto event = Event(clk, port_id, IODirection::In, addr + packetSize * tc_ptr->get_dram_write_offset(id), packetSize, true);
            em_ptr->QueueEvent(event);
            tc_ptr->update_dram_write_offset(id);
            CDCOUT("DRAM::Write to DRAM!" << endl, kDebugLevel);
            dram_tail_flits.push_back(dram_flit);

            // reset dram recv flit 
            dram_recv_flits = 0;
        }
        dram_write_packet.pop_front(); 
      }

      Transaction* tc_ptr = NULL;

      // first remove finished tc
      if (!tc_queue.empty()) {
        tc_ptr = tc_queue.front();   
        if (tc_ptr->finished || 
                (tc_ptr->is_send() && tc_ptr->is_send_finished())) {
            tc_queue.pop_front();
        }   
      }

      // Keep peek_tc, peek will peek the front and pop the front if exists 
      // for unicast/multicast we will get the same tc until all packets for that tc is sent
      // ADD the register to store the currently running tc
      if (tc_queue.empty()) {
         tc_ptr = tc_tracker.peek_tc(id, sc_time_stamp());
         if (tc_ptr != NULL) 
            tc_queue.push_back(tc_ptr);
      } 
      
      if (!tc_queue.empty()) {
        tc_ptr = tc_queue.front();   
      //if (tc_ptr != NULL) {
         //cout << "TC::tc_ptr value " << tc_ptr << endl;
        CDCOUT("Attempt to schedule TC: ", kDebugLevel);
        if (kDebugLevel < 2) { 
            tc_ptr-> print();
        }

        int tc_id = tc_ptr -> get_tc_id();
        int new_print_tc_id = tc_id / 10000;
        if (new_print_tc_id != print_tc_id) {
            cout << "id: " << id << ", tc_id: " << tc_id << ", timestamp: " << sc_time_stamp() <<  endl;
            print_tc_id = new_print_tc_id;
        }

        // if operation is count, wait tc->size number of cycle
        if (tc_ptr->get_op() == OpType::Count) {
            CDCOUT("TC::waited for " << tc_ptr->get_size() << " cycles" << endl, kDebugLevel);
            if (counter_start_time.empty()) {
                counter_start_time.push_back(sc_time_stamp());
            } else {
                //wait(tc_ptr->get_size());  
                // cout <<"Cur "<< sc_time_stamp().to_string() << endl;
                // cout <<"Start"<< counter_start_time.front().to_string() << endl;
                sc_core::sc_time diff = sc_time_stamp() - counter_start_time.front();  
                // cout <<"DIFF VAL"<< diff.value() << endl;
                unsigned int cycle_count = (unsigned int) tc_ptr->get_size();
                // cout <<"DIFF SIZE"<< cycle_count << endl;

                if (diff.value()/1000 >= cycle_count){
                    counter_start_time.pop_front();
                    tc_tracker.rm_tc(tc_ptr, sc_time_stamp()); 
                }

            }
        } else if ((tc_ptr->get_op() == OpType::UniCast) 
                || (tc_ptr->get_op() == OpType::MultiCast)) {

          // tmp read address
          Address addr = 0x30000;
          // make sure all vcs have active packets to push
          for (unsigned i = 0; i < NUM_VCHANNELS; i++) {
            // if packet is empty 
            if (packet[i].empty()) {
              
              dest_per_vc[i].clear();
              int num_flits = tc_ptr -> get_num_flits(); 
              int tc_id = tc_ptr -> get_tc_id();
              RouteType route_type;
              if (tc_ptr->get_op() == OpType::UniCast) route_type = UniCast;
              else route_type = MultiCast;
       
              NVUINTW(DEST_WIDTH) route = 0;

              // generate packet from tc
              tc_ptr->generate_route(route);

              CDCOUT( "TC::generate_packet num_flits: " << num_flits 
                  << ", tc_id: " << tc_id 
                  << ", route_type: " << route_type
                  << ", route: " << route << endl, kDebugLevel);

               
               // should not allow new tc by calling update_sent_size
               // For DRAM node 
               if ((id == DRAMPort) && dram_ren) {
                 // DRAM data bitwidth is 64, BL -> 4 -> 256 bits per dram read request   
                 // which is 32 bytes == each packet size
                 assert(num_flits == spec::noc::maxPktSize);
                 // packetSize in bytes for only data not including header flit
                 unsigned packetSize = (num_flits - 1) * (spec::noc::flitDataWidth >> 3); 
                 // transactionSize in bytes 
                 assert(ddr_ptr->transactionSize == packetSize);
                 
                 // start dram read counter
                 if (tc_ptr->get_dram_read_offset() == 0) {
                    tc_ptr->dram_read_start_cycle = sc_time_stamp();
                 }

                 // for every send, issue DRAM read req
                 // minus one flit for the header, and directly pop the fake packet for header to the dram_read_addrs 
                auto event = Event(clk, port_id, IODirection::In, addr + packetSize * tc_ptr->get_dram_read_offset(), packetSize, false);
                em_ptr->QueueEvent(event);
                tc_ptr->update_dram_read_offset();
                CDCOUT( "DRAM::Read from DRAM!" << endl, kDebugLevel);
                //CDCOUT( "tc_id: " << tc_id << endl, kDebugLevel);
                //CDCOUT( "JEN_DEBUG READ1 dram_read_addrs size: " << dram_read_addrs.size() << endl, kDebugLevel);

                 for (int i = 0; i < num_flits - 1; i++) {
                     // pop fake dram read packet, since each DRAM read is 4 * flits
                     dram_read_addrs.push_back(addr+i*(spec::noc::flitDataWidth >> 3));
                 }
                //CDCOUT( "JEN_DEBUG READ1 dram_read_addrs size: " << dram_read_addrs.size() << endl, kDebugLevel);
               }

              // construct flits for packet
              packet[i] = generate_packet(i, dest_per_vc[i], num_flits, tc_id, route_type, route);
              // update sent_size for packet to be sent
              //cout << "TC::update_sent_size  i:" << i << ", id: " << id << endl;
              //tc_ptr->update_sent_size();
            }
          }

        } else {
            cout << "Unknown TC operation!" << endl;
            assert(false);
        }
      }

      // Pull from DRAM 
      if ((id == DRAMPort) && dram_en()) {
          auto e = em_ptr ->  GetWatchedEvent();  
          while(e.is_valid) {
            CDCOUT( "DRAM::Received event clock:" << e.clock 
              << " port_id:" << e.port_id << " address:"
              << e.address << " is_write:" << e.is_write 
              << " data_size: " << e.data_size << endl, kDebugLevel);

            // Read from DRAM Callback
            if (!e.is_write) {
              // Update DRAM read size 
              tc_ptr->update_dram_read_size();

              //CDCOUT("JEN_DEBUG READ2 tc_id:" << tc_ptr->get_tc_id() << ", dram_read_size: " << tc_ptr->dram_read_size << std::endl, kDebugLevel);
              // Check if it finished loading packets 
              if (tc_ptr->is_dram_read_finished()) {
                  tc_ptr->dram_read_end_cycle = sc_time_stamp();
                  //CDCOUT("JEN_DEBUG FINISH: " << tc_ptr->dram_read_end_cycle << std::endl, kDebugLevel);
              }
              dram_read_addrs.push_back(e.address);
              //CDCOUT( "JEN_DEBUG READ2 dram_read_addrs: " << dram_read_addrs.size() << endl, kDebugLevel);
            // Write to DRAM Callback
            } else {
              assert(!dram_tail_flits.empty());
              Flit_t dram_tail_flit = dram_tail_flits.front(); 
              Transaction* tc_ptr = tc_tracker.get_flit_tc(dram_tail_flit);
              tc_ptr->update_dram_write_size(id); 
              if (tc_ptr->is_dram_write_finished(id)) {
                tc_ptr->dram_write_end_cycle_multi[id] = sc_time_stamp();
              }
              tc_tracker.flit_received(id, dram_tail_flit, sc_time_stamp());
              dram_tail_flits.pop_front();

            } 
            e = em_ptr ->  GetWatchedEvent(); 
          }
      }
      // at this point all VCs should have flits to be sent
      // but not all VCs will have credits
      // randomly select a vc
        out_vc = select_vc();
        if (!dest_per_vc[out_vc].empty()) {
          if (out_vc != -1) {
            if (!packet[out_vc].empty()) {

              if ( (!dram_ren) || (id != DRAMPort) ||
                    ((id == DRAMPort) && !dram_read_addrs.empty())) {
                Flit_t flit = packet[out_vc].front(); // Peek front
                if (out.PushNB(flit)) { // If send is successful
                  packet[out_vc].pop_front(); // Pop

                  if ((id == DRAMPort) && dram_ren) {
                      dram_read_addrs.pop_front(); 
                      //CDCOUT("JEN_DEBUG POP tc_id:" << tc_ptr->get_tc_id() << ", dram_read_addr: " << dram_read_addrs.size() << std::endl, kDebugLevel);
                      //CDCOUT("JEN_DEBUG POP sent_size:" << tc_ptr->get_sent_size() << ", size: " << tc_ptr->get_size() << std::endl, kDebugLevel);
                  }
                  ref.flit_sent(id, dest_per_vc[out_vc], out_vc, flit);
                  // Not use flit_sent, this update to sent_size is too late
                  tc_tracker.flit_sent(id, flit);
                  num_flit += 1;
                  
                } else {
                  CDCOUT( "ID: " << id << " Push Unsuccessful\n", kDebugLevel);
                }
              }
            }
          }
        }

      // Add the pacer delay? 
      //while (pacer.tic())
      //  wait();
      // stop the simulation when tc_tracker finishes tcs in tc_vecs
      if (id == 0) {
        if (tc_tracker.is_finish()) {
          sc_stop();
          tc_tracker.print_tc_timestamps(DRAMPort);
          tc_tracker.print_tc_stats(sc_time_stamp());
          cout << "FINISH @" << sc_time_stamp() << endl;
        }
        auto cur_time = sc_time_stamp().to_double();
        
        // Add time limit to the run
        double limit_ns = 70000000L;
        //double limit_ns = 700L;
        //double limit_ns = 700L;
        double limit_ps = limit_ns * 1000; 
        sc_core::sc_time total_cycles(limit_ps, SC_PS);
        if (cur_time > limit_ps) {
          sc_stop();
          //tc_tracker.print_tc_timestamps(DRAMPort);
          tc_tracker.print_tc_stats(total_cycles);
          cout << "FINISH @" << sc_time_stamp() << endl;
        }

      }

      if ((id == DRAMPort) && dram_en()) {
        em_ptr->ClockTick(clk);
      }
    }
  }

  flits_t generate_packet(int vc, deque<int>& dest, int& num_flits, int& tc_id, RouteType& route_type, NVUINTW(DEST_WIDTH)& route) {
    static NVUINTW(Flit_t::data_width) unique = 0;
    flits_t packet;

    Flit_t flit;

#if (PACKETIDWIDTH > 0)
    flit.packet_id = vc;
#endif
    for (int i = 0; i < num_flits; i++) {

      // Packet id should be unique for every source. It will be used to detect
      // flit at sink
      //flit.data = (++unique);
      flit.data = (0);
      if (num_flits == 1) {
        flit.flit_id.set(
            FlitId2bit::SNGL);  // FIXME: make it grabable from Flit_t
      } else if (i == 0) {
        flit.flit_id.set(FlitId2bit::HEAD);
      } else if (i == num_flits - 1) {
        flit.flit_id.set(FlitId2bit::TAIL);
      } else {
        flit.flit_id.set(FlitId2bit::BODY);
      }

      if (flit.flit_id.isHeader()) {
        //NVUINTW(DEST_WIDTH) route = create_route(num_flits);
        flit.data = (unique << DEST_WIDTH + 1) + route;
                
        flit.data[DEST_WIDTH] = ((route_type == MultiCast) ||
                                 (route_type == MultiCast_SingleSource) ||
                                 (route_type == MultiCast_DualSource));

        if ((route_type == MultiCast) ||
            (route_type == MultiCast_SingleSource) ||
            (route_type == MultiCast_DualSource)) {

            // Assume no 2nd PoC
            NVUINTW(NOC_MCAST_DEST_WIDTH)
              dest_ids = nvhls::get_slc<NOC_MCAST_DEST_WIDTH>(flit.data, NOC2_MCAST_DEST_WIDTH);
            for (int k = 0; k < NOC_MCAST_DEST_WIDTH; k++) {
                if (dest_ids[k] == 1) {
                    int ref_dest_id = k;
                    CDCOUT("ref_dest_id: " << ref_dest_id << " from router " << id << endl, kDebugLevel);
                    dest.push_back(ref_dest_id);
                }
            }
        } else {
            NVUINTW(NOC_UCAST_DEST_WIDTH)
            dest_id = nvhls::get_slc<NOC_UCAST_DEST_WIDTH>(flit.data, 0);

            int ref_dest_id = dest_id;
            CDCOUT("ref_dest_id: " << ref_dest_id << " from router " << id << " port " << i << endl, kDebugLevel);
            dest.push_back(ref_dest_id);
        }
        if ((route_type == MultiCast) ||
            (route_type == MultiCast_SingleSource) ||
            (route_type == MultiCast_DualSource)) {
          assert(dest.size() == 1 || id == 0 || num_flits > 1 ||
                 DEST_WIDTH != Flit_t::data_width);  // only id==0 can write to
                                                     // local ports for single
                                                     // flit packets (and thus
                                                     // create a snigle flit
                                                     // multicast)
        } else {
          assert(dest.size() ==
                 1);  // FIXME - for now on paper should support more
        }

        CDCOUT("###### " << dest.size() << endl, kDebugLevel);
      }

      //cout << "DEST_WIDTH" <<DEST_WIDTH <<endl;
      // Make sure the unique number does not overflow to tc id
       
      if (flit.data >> FLIT_SHIFT_WIDTH != 0){
              cout << "TC::generate_packet num_ uniqueflits: " << num_flits 
                  << ", tc_id: " << tc_id 
                  << ", unique: " << unique 
                  << ", route_type: " << route_type
                  << ", route: " << route << endl;
       }

       
      assert(flit.data >> FLIT_SHIFT_WIDTH == 0);

      // Set top FLIT_TC_WIDTH of the route header to tc_id
      NVUINTW(FLIT_TC_WIDTH) tc_id_bits = tc_id; 
      flit.data = nvhls::set_slc(flit.data, tc_id_bits, FLIT_SHIFT_WIDTH);

      CDCOUT("@" << sc_time_stamp() << " Source ID: " << id
           << " :: Write = " << flit << endl, kDebugLevel);
      if (dest.size() != 0) {
        packet.push_back(flit);
      }
    }
    return packet;
  }

  NVUINTW(DEST_WIDTH) create_route(int num_flits, RouteType route_type) {
    NVUINTW(DEST_WIDTH) route = 0;
    // in case of a single flit packet we must have the "rest" of the bits,
    // left to current hop, to carry some information.
    // otherwise we won't be able to identify the origin of this flit at
    // destination. In case of flits to local ports we
    // are required to zero out that part to prevent recognizing remote port
    // 0 as destination. Therefore the testbench
    // can not cover single flit packets sent to local ports. Given that the
    // internal implementation of router is implemented
    // without distinguishing between local and remote ports after routing
    // decompression, I think it is a tolerable limitation.
    // To ease it a bit, we can allow single flit packets sent to local
    // ports by only one source (id==0), then identification
    // won't be an issue.
    int dest_id;
    route = 0;
    NVUINTW(NOC_MCAST_DEST_WIDTH) dest_ids=0xf; 

    switch (route_type) {
      case UniCast:
        dest_id = rand() % NOC_DEST_WIDTH;
        //dest_id = rand() % kNumNoCMeshLPorts;
        //dest_id = id;
        route = dest_id;
        break;
      case UniCast_NoCongestion:
        route = id;
        break;
      case UniCast_SingleDest:
        route = 1;
        break;
     case MultiCast:
        //route = nvhls::get_rand<NOC_MCAST_DEST_WIDTH>();
        route = nvhls::set_slc(route, dest_ids, 0);
        route <<= NOC2_MCAST_DEST_WIDTH;
        route[0] = 1;  // Indicates local noc2
        break;
      case MultiCast_SingleSource:
        if (id == 1) {
          route = dest_ids;
          //route = nvhls::get_rand<NOC_MCAST_DEST_WIDTH>();
          route <<= NOC2_MCAST_DEST_WIDTH;
          route[0] = 1;  // Indicates local noc2
        } else {
          route = 0;
        }
        break;
      case MultiCast_DualSource:
        if ((id == 1) || (id == 0)) {
          route = nvhls::get_rand<NOC_MCAST_DEST_WIDTH>();
          route <<= NOC2_MCAST_DEST_WIDTH;
          route[0] = 1;  // Indicates local noc2
        } else {
          route = 0;
        }
        break;
      default:
        break;
    }
    return route;
  }

  SC_HAS_PROCESS(SrcDest);

  SrcDest(sc_module_name name_, const unsigned int& id_, const Pacer& pacer_,
         Reference& ref_, TCTracker& tc_tracker_)
      : sc_module(name_),
        out("out"),
        clk("clk"),
        rst("rst"),
        id(id_),
        pacer(pacer_),
        ref(ref_),
        tc_tracker(tc_tracker_) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
};

// utility function to generate variety for probabilities for different
// instances
double rate_gen(int i, double slope, double offset) {
  double linear = slope * i + offset;
  return (linear - static_cast<int>(linear));
}

SC_MODULE(testbench) {
  NVHLS_DESIGN(HybridRouterTop) router;

  typedef Connections::Combinational<Flit_t> DataChan;

  sc_clock clk;
  sc_signal<bool> rst;
  Reference ref;

  string csv_filename;
  string json_filename;
  string stats_filename;
  TCTracker tc_tracker;

  // JTAG for NoC
  sc_signal<HybridRouterTop::NoCRouteLUT_t> NoCRouteLUT_jtags[spec::kNumNoCRouters][NOC_DEST_WIDTH];
  sc_signal<HybridRouterTop::NoC2RouteLUT_t> NoC2RouteLUT_jtags[spec::kNumNoCRouters][NOC2_DEST_WIDTH];
  sc_signal<HybridRouterTop::noc2_id_t> noc2_id_jtags[spec::kNumNoCRouters * spec::kNumNoC2Routers];


  SC_HAS_PROCESS(testbench);
  //SC_CTOR(testbench)
  testbench(sc_module_name name_, 
          string& csv_filename_, 
          string& json_filename_, 
          string& stats_filename_)
      : sc_module(name_),
        router("router"),
        clk("clk", 1.0, SC_NS, 0.5, 0, SC_NS, false),
        rst("rst"), 
        csv_filename(csv_filename_), 
        json_filename(json_filename_), 
        stats_filename(stats_filename_){

    // Init Transaction Tracker
    int num_srcdest = NOC_DEST_WIDTH;

    tc_tracker.init(csv_filename, num_srcdest, json_filename, stats_filename);
    tc_tracker.print();

    Connections::set_sim_clk(&clk);

    router.clk(clk);
    router.rst(rst);

    // Connect JTAG
    // The route table for the dest
    // The index is the dest_id
    int NoCRouteLUT_data[spec::kNumNoCRouters][NOC_DEST_WIDTH];
    //    = {
    //    8, 8, 16, 2, 2, 8, 8, 1};

    for (size_t n = 0;  n < spec::kNumNoCRouters; n++){
        for (int i = 0; i < NOC_DEST_WIDTH; ++i) {
            NoCRouteLUT_data[n][i] = get_port(n, i, true);
        }
    }

    for (size_t n = 0;  n < spec::kNumNoCRouters; n++){
        for (int i = 0; i < NOC_DEST_WIDTH; ++i) {
          router.NoCRouteLUT_jtags[n][i](NoCRouteLUT_jtags[n][i]);
          //router.NoCRouteLUT_jtags[n*NOC_DEST_WIDTH+i](NoCRouteLUT_jtags[n][i]);
          NoCRouteLUT_jtags[n][i].write(NoCRouteLUT_data[n][i]);
        }
    }

    // Always going to the left to go to NoP
    const int NoC2RouteLUT_data[NOC2_DEST_WIDTH] = {0x00}; 
    for (size_t n = 0;  n < spec::kNumNoCRouters; n++){
        for (int i = 0; i < NOC2_DEST_WIDTH; ++i) {
          router.NoC2RouteLUT_jtags[n][i](NoC2RouteLUT_jtags[n][i]);
          //router.NoC2RouteLUT_jtags[n*NOC2_DEST_WIDTH+i](NoC2RouteLUT_jtags[n][i]);
          NoC2RouteLUT_jtags[n][i].write(NoC2RouteLUT_data[i]);
        }
        router.noc2_id_jtags[n](noc2_id_jtags[n]);
        noc2_id_jtags[n].write(0);
    }
    //for (int i = 0; i < HybridRouterTop::NumPorts; ++i) {
    for (int i = 0; i < HybridRouterTop::kNumNoCMeshPorts; ++i) {
      {  // attach a source
        // Create and attach SOURCE where expected by config table
        // FIXME: I don't care about never destructing these
        ostringstream ss;
        ss << "srcdest_on_port_" << i;
        SrcDest* srcdest= new SrcDest(
            ss.str().c_str(), i,
            Pacer(rate_gen(i, 0.2, 0.2), rate_gen(i, 0.3, 0.8)), ref, tc_tracker);
        DataChan* data_chan0 = new DataChan();
        DataChan* data_chan1 = new DataChan();
        srcdest->clk(clk);
        srcdest->rst(rst);
        srcdest->out(*data_chan0);
        router.in_port[i](*data_chan0);
        srcdest->in(*data_chan1);
        router.out_port[i](*data_chan1);
      }
    }

    SC_THREAD(run);
  }

  void run() {
    // reset
    rst = 0;
    cout << "@" << sc_time_stamp() << " Asserting Reset " << endl;
    wait(2, SC_NS);
    cout << "@" << sc_time_stamp() << " Deasserting Reset " << endl;
    rst = 1;
    wait(1000000000, SC_NS);
    cout << "@" << sc_time_stamp() << " Stop " << endl;
    sc_stop();
  }
};

int sc_main(int argc, char* argv[]) { 
  string csv_filename;
  string json_filename;
  string stats_filename;

  if (argc == 1) {
    csv_filename = "tc.csv";
    json_filename = "tc.json";
    stats_filename = "tc.summary.json";
  } else if (argc == 2) {
    string filename_prefix = argv[1];
    csv_filename = filename_prefix + ".csv";
    json_filename = filename_prefix + ".json";
    stats_filename = filename_prefix + ".summary.json";
  } else {
    cout << "Please specify input csv filename and output json filename!" << endl;
    assert(false);
  }
  
  std::string DRAM_PORT_str = getEnvVar("DRAM_PORT");
  std::cout << "DRAM_PORT: " << DRAM_PORT_str << std::endl;
  if (DRAM_PORT_str != "") {
      DRAMPort = (size_t) std::stoi(DRAM_PORT_str);
  }
  std::cout << "DRAMPort: " << DRAMPort << std::endl;
  std::cout << "kNumNoCMeshPorts: " <<  kNumNoCMeshPorts << std::endl;
  assert(DRAMPort < kNumNoCMeshPorts);

  nvhls::set_random_seed();
  testbench my_testbench("my_testbench", csv_filename, json_filename, stats_filename);
  sc_report_handler::set_actions(SC_ERROR, SC_DISPLAY);
  sc_start();
  bool rc = (sc_report_handler::get_count(SC_ERROR) > 0);

  if (my_testbench.ref.send_cnt != my_testbench.ref.recv_cnt) {
    rc = true;
  }

  if (rc)
    cout << "Simulation FAILED\n";
  else
    cout << "Simulation PASSED\n";
  return rc;
};
