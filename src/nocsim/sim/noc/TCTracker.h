#ifndef __TCTRACKER_H__
#define __TCTRACKER_H__
#include "Transaction.h"
#include <systemc.h>
#include <mc_scverify.h>
#include <testbench/Pacer.h>
#include <testbench/nvhls_rand.h>
#include <nvhls_connections.h>
#include "HybridRouterTop.h"
#include "RouterSpec.h"

const int FLIT_DATA_WIDTH = spec::noc::flitDataWidth;
const int FLIT_TC_WIDTH = spec::noc::flitTCWidth;
const int FLIT_SHIFT_WIDTH = FLIT_DATA_WIDTH - FLIT_TC_WIDTH;

typedef HybridRouterTop::Flit_t Flit_t;
typedef std::deque<Flit_t> flits_t;

class TCTracker
{
    // this struct is used for destructer
    std::vector<Transaction*> tcs;

    // this struct is used for easy tc lookup 
    std::map<int, Transaction*> tc_id_map;

    // this stores kNumNoCMeshPorts of vectors of transactions
    // with each transaction's actor being the pe id
    std::vector<std::vector<Transaction*> > tc_vecs;

    std::string json_filename; 
    std::string stats_filename; 

    // tc_id of received headers 
    // check if it is possible to have interleaved recv and send
    std::vector<std::vector<int> > recv_headers; 
    std::vector<std::vector<int> > sent_headers; 

    
    // stores the total # of cycle for running each op
    std::map<int, double> op_stats; 

    public: 
        TCTracker(){};
        ~TCTracker();
        void init(std::string& filename, int num_srcdest, 
            std::string& json_fn, std::string& stats_fn);
        void print();

        // flit_received
        void rm_dep_on_pe(int tc_id, int pe_id); 
        void rm_dep(int tc_id); 
        void rm_tc(Transaction* tc_ptr, sc_core::sc_time timestamp);
        void flit_received(const unsigned int& id, const Flit_t& flit, sc_core::sc_time timestamp);
        bool flit_sent(const unsigned int& id, const Flit_t& flit);
        Transaction* get_flit_tc(const Flit_t& flit);
  
        // flit_sent
        Transaction* peek_tc(const unsigned int& id, sc_core::sc_time timestamp);
        
        // check if finish
        bool is_finish();

        // print timestamp of each tc 
        void print_tc_timestamps(int DRAMPort);
        void print_tc_stats(sc_core::sc_time total_cycles);
};

void TCTracker::print_tc_stats(sc_core::sc_time total_cycles) {
    std::ofstream prof_log;     
    prof_log.open(this->stats_filename.c_str());
    prof_log << "{\"total_cycles\":" << total_cycles.to_double() / 1000; 
    for(std::map<int, double>::iterator it = op_stats.begin(); it != op_stats.end(); ++it) { 
        prof_log << ", "<<std::endl;
        prof_log << "\"" << Transaction::print_op(it->first) << "_cycles\":" << (long long unsigned) it->second; 
    }
    prof_log << "}" << std::endl;
    prof_log.close();
}

// print the transaction for visualization
void TCTracker::print_tc_timestamps(int DRAMPort) {
    std::ofstream prof_log; 
    prof_log.open(this->json_filename.c_str());
    
    // TODO Make overlapping TC new thread
    //tids = {}
    prof_log << "[" << std::endl;
    for (std::vector<Transaction*>::iterator it = tcs.begin(); it != tcs.end(); ++it) {
        Transaction* tc_ptr = *it; 
        //{"pid":375,"tid":775,"ts":705828478699,"ph":"E","cat":"sequence_manager","name":"RunNormalPriorityTask","tts":65756164252,"args":{}},
        int id = tc_ptr->get_actor_id();  
        int category = 0;
        std::string op_name = Transaction::print_op(tc_ptr->get_op());
        int op_id = tc_ptr->get_op(); 

        if (op_id == OpType::MultiCast || op_id == OpType::UniCast) { 
          // Print DRAM read cycles
          if (id == DRAMPort) {
                // hardcoded pid == 31 
                 prof_log << "{\"pid\":" << id << ",\"tid\":" << 1 << ",\"ts\":"
                    << (unsigned long long)(tc_ptr->dram_read_start_cycle.to_double()/ 1000)
                    <<",\"ph\":\"B\",\"cat\":\"Node_"
                    << category << "\",\"name\":\"TC_" << tc_ptr->get_tc_id() 
                    << "_" << op_name 
                    << "-DRAM-Read" 
                    << "\",\"args\":{}}," 
                    << std::endl;
                 prof_log << "{\"pid\":" << id << ",\"tid\":" << 1 << ",\"ts\":"
                    << (unsigned long long)(tc_ptr->dram_read_end_cycle.to_double()/ 1000)
                    <<",\"ph\":\"E\",\"cat\":\"Node_"
                    << category << "\",\"name\":\"TC_" << tc_ptr->get_tc_id() 
                    << "_" << op_name 
                    << "-DRAM-Read" 
                    << "\",\"args\":{}}," 
                    << std::endl;

                double dur = tc_ptr->dram_read_end_cycle.to_double()/ 1000 - tc_ptr->dram_read_start_cycle.to_double()/ 1000; 
                op_stats[3] += dur; 
            }
          // Print DRAM write cycles 
          if(tc_ptr->is_id_dest(DRAMPort)) {
                // hardcoded pid == 31 
                 prof_log << "{\"pid\":" << id << ",\"tid\":" << 1 << ",\"ts\":"
                    << (unsigned long long)(tc_ptr->dram_write_start_cycle_multi[DRAMPort].to_double()/ 1000)
                    <<",\"ph\":\"B\",\"cat\":\"Node_"
                    << category << "\",\"name\":\"TC_" << tc_ptr->get_tc_id() 
                    << "_" << op_name 
                    << "-DRAM-Write" 
                    << "\",\"args\":{}}," 
                    << std::endl;
                 prof_log << "{\"pid\":" << id << ",\"tid\":" << 1 << ",\"ts\":"
                    << (unsigned long long)(tc_ptr->dram_write_end_cycle_multi[DRAMPort].to_double()/ 1000)
                    <<",\"ph\":\"E\",\"cat\":\"Node_"
                    << category << "\",\"name\":\"TC_" << tc_ptr->get_tc_id() 
                    << "_" << op_name 
                    << "-DRAM-Write" 
                    << "\",\"args\":{}}," 
                    << std::endl;
                double dur = tc_ptr->dram_write_end_cycle_multi[DRAMPort].to_double()/ 1000 - tc_ptr->dram_write_start_cycle_multi[DRAMPort].to_double()/ 1000; 
                op_stats[4] += dur; 
          }  

        }
        if (tc_ptr->get_op() != OpType::MultiCast) { // if not multicast
            prof_log << "{\"pid\":" << id << ",\"tid\":" << 0 << ",\"ts\":"
                << (unsigned long long)(tc_ptr->start_cycle.to_double()/ 1000)
                <<",\"ph\":\"B\",\"cat\":\"Node_"
                << category << "\",\"name\":\"TC_" << tc_ptr->get_tc_id() 
                << "_" << op_name << "\",\"args\":{}}," 
                << std::endl;

            prof_log << "{\"pid\":" << id << ",\"tid\":" << 0 << ",\"ts\":"
                << (unsigned long long)(tc_ptr->end_cycle.to_double()/ 1000) 
                <<",\"ph\":\"E\",\"cat\":\"Node_"
                << category << "\",\"name\":\"TC_" << tc_ptr->get_tc_id() 
                << "_" << op_name << "\",\"args\":{}}"; 
            
            if( it+1 != tcs.end()) prof_log << ",";
            prof_log << std::endl;

            // TODO should update op_stats and we ran the simulation
            // This is a bit of a hack
            double dur = tc_ptr->end_cycle.to_double()/ 1000 - tc_ptr->start_cycle.to_double()/ 1000;
            assert(dur >= 0.0);
            op_stats[tc_ptr->get_op()] += dur; 
        } else { 
            for (std::vector<int>::iterator dest_it = tc_ptr->dest.begin(); dest_it != tc_ptr->dest.end(); ++dest_it) {
                prof_log << "{\"pid\":" << id << ",\"tid\":" << (*dest_it) + 2 << ",\"ts\":"
                    << (unsigned long long)(tc_ptr->start_cycle.to_double()/ 1000)
                    <<",\"ph\":\"B\",\"cat\":\"Node_"
                    << category << "\",\"name\":\"TC_" << tc_ptr->get_tc_id() 
                    << "_" << op_name 
                    << "_" << *dest_it
                    << "\",\"args\":{}}," 
                    << std::endl;

                prof_log << "{\"pid\":" << id << ",\"tid\":" << (*dest_it) + 2 << ",\"ts\":"
                    << (unsigned long long)(tc_ptr->end_cycle_multi[*dest_it].to_double()/ 1000) 
                    <<",\"ph\":\"E\",\"cat\":\"Node_"
                    << category << "\",\"name\":\"TC_" << tc_ptr->get_tc_id() 
                    << "_" << op_name 
                    << "_" << *dest_it
                    << "\",\"args\":{}}"; 

                if( it+1 != tcs.end()) prof_log << ",";
                prof_log << std::endl;

                double dur = tc_ptr->end_cycle.to_double()/ 1000 - tc_ptr->start_cycle.to_double()/ 1000;
                assert(dur >= 0.0);
                op_stats[tc_ptr->get_op()] += dur; 
            }
        }
    }
    prof_log << "]" << std::endl;
    prof_log.close();
}

// init TCTracker from .csv file
void TCTracker::init(std::string& filename, int num_srcdest,
                        std::string& json_fn, std::string& stats_fn) {
      
    
    this->json_filename.assign(json_fn);
    this->stats_filename.assign(stats_fn);
    std::ifstream in(filename.c_str());
    if (!in.is_open()) {
        std::cout << "Cannot open file "  << filename << "!" << std::endl;  
        assert(false); 
    }

    std::string line;
    int tc_id = -1;
    while (std::getline(in,line)) {
        
        // omit commented lines
        if (line.empty() || (line[0] == '#')) {
            continue;       
        }
        Transaction* tc = new Transaction(line); 

        // make sure new tc_id is larger than the previous one
        int new_tc_id = tc->get_tc_id();
        assert(new_tc_id > tc_id);
        // make sure tc_id does not overflow 
        assert(new_tc_id < (2 << FLIT_TC_WIDTH));
        tc_id = new_tc_id;
        tcs.push_back(tc);
        tc_id_map.insert(std::pair<int, Transaction*>(tc_id, tc));
    }
    
    // init the tc_vecs 
    tc_vecs.resize(num_srcdest);
    for (std::vector<Transaction*>::iterator it = tcs.begin(); it != tcs.end(); ++it) {
        int actor_id = (*it)->get_actor_id(); 
        tc_vecs[actor_id].push_back(*it);
    }

    // init recv_headers tracker
    recv_headers.resize(num_srcdest);

    // init op_stats
    op_stats[OpType::UniCast] = 0.0; 
    op_stats[OpType::MultiCast] = 0.0; 
    op_stats[OpType::Count] = 0.0; 
    // Hardcoded DRAM Cycles
    op_stats[3] = 0.0; 
    op_stats[4] = 0.0; 
}

TCTracker::~TCTracker() {
    for (std::vector<Transaction*>::iterator it = tcs.begin(); it != tcs.end(); ++it) {
        delete *it;
    }
}

void TCTracker::print() {
    for (std::vector<std::vector<Transaction*> >::iterator it = tc_vecs.begin(); it != tc_vecs.end(); ++it) {
        CDCOUT("Actor: " << it - tc_vecs.begin() << std::endl, kDebugLevel); 
        CDCOUT("----------------------" << std::endl, kDebugLevel);
        for (std::vector<Transaction*>::iterator it_vec = it->begin(); it_vec != it->end(); ++it_vec) {
            (*it_vec)->print();
        }
        CDCOUT("----------------------" << std::endl, kDebugLevel);
    }
}


void TCTracker::rm_dep_on_pe(int tc_id, int pe_id){ 
    std::vector<Transaction*> vec = tc_vecs[pe_id]; 
    for (std::vector<Transaction*>::iterator it_vec = vec.begin(); it_vec != vec.end(); ++it_vec) {
            Transaction* tc = *it_vec; 
            tc->rm_dep(tc_id);
    }
}


// remove all finished tc_id dep from pending tcs
void TCTracker::rm_dep(int tc_id){ 
    for (std::vector<std::vector<Transaction*> >::iterator it = tc_vecs.begin(); it != tc_vecs.end(); ++it) {
        std::vector<Transaction*> vec = *it; 
        for (std::vector<Transaction*>::iterator it_vec = vec.begin(); it_vec != vec.end(); ++it_vec) {
            Transaction* tc = *it_vec; 
            tc->rm_dep(tc_id);
        }
    }
}

// remove the tc with tc_ptr 
void TCTracker::rm_tc(Transaction* tc_ptr, sc_core::sc_time timestamp) {
    tc_ptr->end_cycle = timestamp;
    tc_ptr->finished = true;
    rm_dep(tc_ptr->get_tc_id());
    std::vector<Transaction*>& vec = tc_vecs[tc_ptr->get_actor_id()];
    vec.erase(std::remove(vec.begin(), vec.end(), tc_ptr), vec.end());
}

// return the next unblocked tc  
Transaction* TCTracker::peek_tc(const unsigned int& id, sc_core::sc_time timestamp) {
    Transaction* ret = NULL;

    std::vector<Transaction*> vec = tc_vecs[id];
    
    for (std::vector<Transaction*>::iterator it = vec.begin(); it != vec.end(); ++it) {
        Transaction* tc = *it; 
        // only peckets with op UniCast and MultiCast is_send_finished will be true
        CDCOUT ("TC::pop_tc source: " <<  id << ", tc_id: " << tc->get_tc_id() 
            << ", size: " << tc->get_size()  << ", sent_size:" << tc->get_sent_size() 
            << ", has_no_dep: " << tc->has_no_dep() 
            << ", is_send_finished: " << tc->is_send_finished()
            << endl, kDebugLevel) ;     

        if (tc->has_no_dep() && !tc->started) {
           // if (tc->is_send()) { 
           //     // If this is the first send, record the time
           //     if (!tc->started) { 
           //         tc->start_cycle = timestamp;
           //         tc->started = true;
           //     }
           //     if (!(tc->is_send_finished())) 
           //         return tc;
           // } else {
                tc->started = true;
                tc->start_cycle = timestamp;
                return tc;
           // }
        }
    }
    return ret;
}

// return the actor tc of the flit
Transaction* TCTracker::get_flit_tc(const Flit_t& flit) {
    int recv_tc_id = to_sc(flit.data >> FLIT_SHIFT_WIDTH).to_uint64(); 
    Transaction* tc_ptr = tc_id_map[recv_tc_id];
    return tc_ptr; 
}

// change the recv_size to reflect the number of packets received
// remove the dep in the tc_vec, if recv_size == size 
void TCTracker::flit_received(const unsigned int& id, const Flit_t& flit, sc_core::sc_time timestamp) {
    
    int recv_tc_id = to_sc(flit.data >> FLIT_SHIFT_WIDTH).to_uint64(); 
    Transaction* tc_ptr = tc_id_map[recv_tc_id];
    CDCOUT("TC::flit_received recv_tc_id: " << recv_tc_id 
      << ", tc_ptr: " << tc_ptr 
      << ", is_recv_finished: " <<  tc_ptr->is_recv_finished()
      << endl, kDebugLevel); 

    // keep track of the headers 
    if (flit.flit_id.isHeader()) {
        // recv_headers[id].push_back(recv_tc_id);
        if (flit.flit_id.isSingle()) {
            // update tc once we receive a packet
            tc_ptr->update_recv_size(id); 
        }
    } else if (flit.flit_id.isTail()){
            tc_ptr->update_recv_size(id); 
    }

    // update the deps of the receiver for multicast
    std::vector<int> ids = tc_ptr->recv_finished_ids(timestamp); 
    for (std::vector<int>::iterator it = ids.begin(); it != ids.end(); ++it) {
       rm_dep_on_pe(tc_ptr->get_tc_id(), *it);  
    }

    // removing the tc from the tc_vecs if recv_size == size
    if (tc_ptr->is_recv_finished()) {
        rm_tc(tc_ptr, timestamp);
    }
}

bool TCTracker::is_finish() {
    int finish = true;
    for (std::vector<std::vector<Transaction*> >::iterator it = tc_vecs.begin(); it != tc_vecs.end(); ++it) {
        std::vector<Transaction*> vec = *it;
        if (vec.size() != 0) finish = false;
    }
    return finish;
}

// Return true if this is a tail packet, else return false
bool TCTracker::flit_sent(const unsigned int& id, const Flit_t& flit) {
    int sent_tc_id = to_sc(flit.data >> FLIT_SHIFT_WIDTH).to_uint64(); 
    Transaction* tc_ptr = tc_id_map[sent_tc_id];

    // keep track of the headers 
    if (flit.flit_id.isHeader()) {
        if (flit.flit_id.isSingle()) {
            // update send packet size    
            tc_ptr->update_sent_size(); 
            // update tc once we receive a packet
            CDCOUT("TC::flit_sent source: " <<  id << ", tc_id: " 
                << sent_tc_id << ", sent_size: " << tc_ptr->get_sent_size() <<endl, kDebugLevel);     
            return true;
        } 
        // else {
        //    sent_headers[id].push_back(sent_tc_id);
        //}
    } else if (flit.flit_id.isTail()) {
        // update send packet size    
        tc_ptr->update_sent_size(); 
        CDCOUT("TC::flit_sent source: " <<  id << ", tc_id: " 
            << sent_tc_id << ", sent_size: " << tc_ptr->get_sent_size() <<endl, kDebugLevel);     
        return true;
        //sent_headers[id].erase(std::remove(sent_headers[id].begin(), 
        //  sent_headers[id].end(), sent_tc_id), sent_headers[id].end());
    }
    return false;
}

#endif
