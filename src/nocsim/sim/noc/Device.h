#ifndef _DEVICE_HPP_
#define _DEVICE_HPP_

#include <map>
#include <Events.h>
#include <DRAMSim.h>
#include <Callback.h>

template <typename T> s_ptr<T> GetChildTyped(s_ptr<Block> block, const std::string &name);

class Memory: public Block {
public:
    Memory(
        const std::string &name,
        s_ptr<Block> parent = nullptr,
        int num_ports=0,
        int line_size=0,
        int size=0);
};

class Bus: public Block {
public:
    int line_size;

    Bus(
        const std::string &name,
        s_ptr<Block> parent,
        int line_size);

    bool ConnectBlock(s_ptr<Block> block, int busPort);
};

class Logic: public Block {
public:
    Logic(const std::string &name, s_ptr<Block> parent);
};

class SRam : public Memory {
public:
    SRam(
        const std::string &name,
        s_ptr<Block> parent = nullptr,
        int num_ports=0,
        int line_size=0,
        int size=0);
};

class CrossBar: public Logic {
public:
    CrossBar(
        const std::string &name,
        s_ptr<Block> parent = nullptr,
        int num_ports = 0,
        int line_size = 0);
};

class DMA: public Logic {
public:
    int line_size;

    DMA(
        const std::string &name,
        s_ptr<Block> parent,
        int line_size = 0);
};

class DDR: public Memory {
public:
    DRAMSim::MultiChannelMemorySystem *mem;
    DRAMSim::TransactionCompleteCB *read_cb;
    DRAMSim::TransactionCompleteCB *write_cb;
    unsigned transactionSize; 

    DDR(
        const std::string &name,
        s_ptr<Block> parent = nullptr,
        int num_ports = 0,
        int line_size = 0,
        int size = 0);

    bool ProcessEvent(Event event) override;
    void ClockTick(Clock clock) override;

    /* callback functors */
    void ReadCompleteCallback(unsigned id, uint64_t address, uint64_t clock_cycle);
    void WriteCompleteCallback(unsigned id, uint64_t address, uint64_t clock_cycle);

    int ExampleTransactions(uint64_t addr);
};


#endif // _DEVICE_HPP_
