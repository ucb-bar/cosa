#ifndef __TRANSACTION_H__
#define __TRANSACTION_H__
#include <boost/tokenizer.hpp>
#include <nvhls_connections.h>
#include "HybridRouterTop.h"
#include "RouterSpec.h"

// To enable debug print
#include <hls_globals.h> 
static const int kDebugLevel = 3;

namespace OpType{
    enum OpType {
      UniCast = 0,
      MultiCast = 1,
      Count = 2,
      DRAM_Read = 3,
      DRAM_Write = 4,
    };
}
// Defines transaction for testbench
class Transaction 
{
    private:
		// tc_id -- transaction id
        // actor_id -- current actor id
        // op -- operation: 0 for unicast, 1 for multicast, 2 for artificial cal with counter
        // size -- for op 0 and 1, the number of packets; for op 2, the number of cycle counts
        int tc_id;
        int actor_id;
        int op; 
        int size; 

        // sent_size -- internal count of sent packets
        int sent_size;
        // recv_size -- internal count of recv packets for unicast
        // int recv_size;
        // recv_size_multi -- internal count of recv packets
        int recv_size_multi[HybridRouterTop::kNumNoCMeshPorts];

        // default num_flits to the max multicast size
        int num_flits;

    public: 
        // src -- list of src
        // dest -- list of dest 
        // dep -- list of dependent tc_id
        std::vector<int> src; 
        std::vector<int> dest;
        std::vector<int> dep;

        // processed 
        bool started;
        bool finished; 
        //bool started_dram_read;
        //bool started_dram_write[HybridRouterTop::kNumNoCMeshPorts];
        // Record the timestamp
        // Including the mem read time for unicast and multicast
        sc_core::sc_time start_cycle;
        sc_core::sc_time end_cycle;

        // Record the timestamp for unicast and multicast mem access time
        sc_core::sc_time dram_read_start_cycle;
        sc_core::sc_time dram_read_end_cycle;

           
        // For multicast with different send finish  
        sc_core::sc_time dram_write_start_cycle_multi[HybridRouterTop::kNumNoCMeshPorts];
        sc_core::sc_time dram_write_end_cycle_multi[HybridRouterTop::kNumNoCMeshPorts];
        sc_core::sc_time end_cycle_multi[HybridRouterTop::kNumNoCMeshPorts];


        // dram read offset for each 32 bytes transaction
        int dram_read_offset;
        // dram read size for each finished 32 bytes transaction
        int dram_read_size;

        // dram write offset for each 32 bytes transaction
        // on the receiver end
        int dram_write_offset[HybridRouterTop::kNumNoCMeshPorts];
        // dram write size for each finished 32 bytes transaction
        int dram_write_size[HybridRouterTop::kNumNoCMeshPorts];

        
        //Default Constructor 
        Transaction(std::string line);
        static int stoi(std::string s); 
        static void stov(std::string s, std::vector<int>& v);

        static std::string print_op(int op);
        void print();

        int get_actor_id(){ return actor_id; }
        int get_tc_id(){ return tc_id; }
        int get_op(){ return op; }
        int get_size() { return size; }
        void set_num_flits(int num) {num_flits = num; }
        int get_num_flits() { return num_flits; } 
        int get_sent_size() { return sent_size; } 
        int get_recv_size(int dest_id) { return recv_size_multi[dest_id]; } 
        int get_dram_read_offset() { return dram_read_offset; } 
        int get_dram_write_offset(int dest_id) { return dram_write_offset[dest_id]; } 
        void update_recv_size(int dest_id){ recv_size_multi[dest_id] += 1; }
        void update_sent_size(){ sent_size += 1; }
        void update_dram_read_offset(){ dram_read_offset += 1; }
        void update_dram_read_size(){ dram_read_size += 1; }
        bool is_dram_read_finished() {return (dram_read_size == size);} 
        void update_dram_write_offset(int dest_id){ dram_write_offset[dest_id] += 1; }
        void update_dram_write_size(int dest_id){ dram_write_size[dest_id] += 1; }
        bool is_dram_write_finished(int dest_id) {return (dram_write_size[dest_id] == size);} 
        bool is_id_dest(int id) {
            for (std::vector<int>::iterator it = dest.begin(); it != dest.end(); ++it) {
                if(id == *it) return true;
            }
            return false;
        }

        bool is_send() {return ((op == OpType::UniCast) || (op == OpType::MultiCast)); }
        bool is_send_finished(){ 
            return (sent_size == size); }

        bool is_recv_finished(){ 
            bool finished = true;
            
            for (int i = 0; i <  HybridRouterTop::kNumNoCMeshPorts; i++) {
                //cout << i  << ": " << recv_size_multi[i] << ", ";
                if ((recv_size_multi[i] != -1) && (recv_size_multi[i] != size)) {
                    finished = false;
                }
            }
            //cout << endl;
            return finished; 
        }          


        // Return the recv ids and reset the recv_size_multi[i] to -1 to 
        // indicate there is no more incoming data
        std::vector<int> recv_finished_ids(sc_core::sc_time timestamp){ 
            std::vector<int> ids; 
            for (int i = 0; i <  HybridRouterTop::kNumNoCMeshPorts; i++) {
                if (recv_size_multi[i] == size) {
                    ids.push_back(i);
                    recv_size_multi[i] = -1;
                    end_cycle_multi[i] = timestamp;
                }
            }
            return ids; 
        }          


        bool has_no_dep(){ return (dep.size() == 0);}
        // remove the dependency on tc_id
        void rm_dep(int tc_id) {
            dep.erase(std::remove(dep.begin(), dep.end(), tc_id), dep.end());
        }
        void generate_route(NVUINTW(HybridRouterTop::DEST_WIDTH)& route);

        //Default Destructor 
        ~Transaction()
        {
            ;
        }
};

Transaction::Transaction(std::string line) {
    typedef boost::tokenizer< boost::escaped_list_separator<char> > tokenizer;
    std::vector<std::string> vec;
    tokenizer tok(line);
    vec.assign(tok.begin(),tok.end());

    // vector now contains strings from one row, output to cout here
    //copy(vec.begin(), vec.end(), std::ostream_iterator<std::string>(cout, "|"));
    //cout << "\n----------------------" << endl;

    tc_id = Transaction::stoi(vec[0]);  
    actor_id = Transaction::stoi(vec[1]);  
    assert(actor_id >= 0 && actor_id < HybridRouterTop::kNumNoCMeshPorts);
    op = Transaction::stoi(vec[2]);  
    size = Transaction::stoi(vec[3]);
    assert(size > 0);
    dram_read_offset = 0;
    started = false;
    finished = false;
    //started_dram_read = false;

    sent_size = 0;
    dram_read_size = 0; 
    // -1 means not initialized
    for (int i = 0; i < HybridRouterTop::kNumNoCMeshPorts; i++) { 
        //started_dram_write[i] = false;
        recv_size_multi[i] = -1;
        dram_write_offset[i] = 0;
        dram_write_size[i] = 0;
    }
    num_flits = spec::noc::maxPktSize;
    
    Transaction::stov(vec[4], src);
    Transaction::stov(vec[5], dest);
    Transaction::stov(vec[6], dep);
    // Currently assume actor_id == src[0]
    // set the 0 for all dests for multicast
    std::cout << line << std::endl;
    for (std::vector<int>::iterator it = dest.begin(); it != dest.end(); ++it) {
        cout << *it << " ";
        cout << HybridRouterTop::kNumNoCMeshPorts << " ";
        assert(*it < HybridRouterTop::kNumNoCMeshPorts);
        recv_size_multi[*it] = 0;
    }
    //for (std::vector<int>::iterator it = src.begin(); it != src.end(); ++it) {
    //    std::cout << *it << " ";
    //}
    //std::cout << std::endl;
    //std::cout << src.data() << std::endl;
    //std::cout << src.size() << std::endl;
    assert(src.size() < 2);
    if (src.size() == 1) {
        assert(actor_id == src[0]);
    }
    
}  

std::string Transaction::print_op(int op) {
    OpType::OpType op_type = (OpType::OpType)op;
    std::string str; 
    switch (op_type) {
        case OpType::UniCast:
            str = "UniCast"; 
            break;
        case OpType::MultiCast:
            str = "MultiCast"; 
            break;
        case OpType::Count:
            str = "Count"; 
            break;
        case OpType::DRAM_Read:
            str = "DRAM_Read"; 
            break;
        case OpType::DRAM_Write:
            str = "DRAM_Write"; 
            break;
        default:
            break;
    }
    return str;
}

// print tc
void Transaction::print() {
    cout << "tc_id: " << tc_id << ", "
        << "actor_id: " << actor_id << ", "
        << "op: " << print_op(op) << ", "
        << "src: ";
    copy(src.begin(), src.end(), std::ostream_iterator<int>(cout, " "));
    cout << "dest: ";
    copy(dest.begin(), dest.end(), std::ostream_iterator<int>(cout, " "));
    cout << "dep: ";
    copy(dep.begin(), dep.end(), std::ostream_iterator<int>(cout, " "));
    cout << endl;
}

// string to int parser 
int Transaction::stoi(std::string s) { 
    try {
        int i = std::atoi(s.c_str());
        //std::cout << i << '\n';
        return i;
    } catch (std::invalid_argument const &e) {
        std::cout << "Bad input: std::invalid_argument thrown" << '\n';
        assert(false);
    } catch (std::out_of_range const &e) {
        std::cout << "Integer overflow: std::out_of_range thrown" << '\n';
        assert(false);
    } 
}

// string to int vector 
void Transaction::stov(std::string s, std::vector<int>& v){
    typedef boost::tokenizer<boost::char_separator<char> > 
        tokenizer;
    boost::char_separator<char> sep(" ");
    std::vector<std::string> vec;
    tokenizer tok(s, sep);
    vec.assign(tok.begin(),tok.end());
    for (std::vector<std::string>::iterator it = vec.begin(); it != vec.end(); ++it) {
        v.push_back(Transaction::stoi(*it)); 
    }
}

// set up packet format
void Transaction::generate_route(NVUINTW(HybridRouterTop::DEST_WIDTH)& route) {
    route = 0;
    int dest_id;
    OpType::OpType op_type = (OpType::OpType)op;
    switch (op_type) {
        case OpType::UniCast:
            assert(dest.size() == 1);
            dest_id = dest[0]; 
            route = dest_id;
            break;
        case OpType::MultiCast:
            {
            // Always at NoP 1 for now
            NVUINTW(HybridRouterTop::NOC2_MCAST_DEST_WIDTH) nop_id = 1;
            route = nvhls::set_slc(route, nop_id, 0);
            for (std::vector<int>::iterator it = dest.begin(); it != dest.end(); ++it) {
                dest_id = *it; 
                assert(dest_id < HybridRouterTop::NOC_MCAST_DEST_WIDTH); 
                route[dest_id + HybridRouterTop::NOC2_MCAST_DEST_WIDTH] = 1;
            }
            }
            break;
        default:
            break;
    }
}

#endif
