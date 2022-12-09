#include <iostream>
#include <Device.h>
#include <SystemConfiguration.h>

template <typename To, typename From>
inline std::shared_ptr<To> reinterpret_pointer_cast(
    std::shared_ptr<From> const & ptr) noexcept
{ return std::shared_ptr<To>(ptr, reinterpret_cast<To *>(ptr.get())); }

template <typename T> s_ptr<T> GetChildTyped(
        s_ptr<Block> block,
        const std::string &name) {
    return reinterpret_pointer_cast<T, Block>(block->children[name]);
}

template s_ptr<Bus> GetChildTyped<Bus>(s_ptr<Block> block, const std::string &name);
template s_ptr<CrossBar> GetChildTyped<CrossBar>(s_ptr<Block> block, const std::string &name);

Memory::Memory(
    const std::string &name, s_ptr<Block> parent,
    int num_ports, int line_size, int size)
    : Block(name, parent) {
    for(int i = 0; i<num_ports; i++) {
        auto s = std::to_string(i);
        ports[s] = em()->CreatePort(s, this);
    }
}

Bus::Bus(const std::string &name, s_ptr<Block> parent, int line_size)
    : Block(name, parent), line_size(line_size) {
    ports["0"] = em()->CreatePort("0", this);
    ports["1"] = em()->CreatePort("1", this);
}

Logic::Logic(const std::string &name, s_ptr<Block> parent)
    : Block(name, parent) {

}

SRam::SRam(
    const std::string &name, s_ptr<Block> parent,
    int num_ports, int line_size, int size)
    : Memory(name, parent, num_ports, line_size, size) {
}

CrossBar::CrossBar(
    const std::string &name, s_ptr<Block> parent,
    int num_ports, int line_size)
    : Logic(name, parent) {

    for(int i = 0; i<num_ports; i++) {
        auto t = "T" + std::to_string(i);
        ports[t] = em()->CreatePort(t, this);

        auto b = "B" + std::to_string(i);
        ports[b] = em()->CreatePort(b, this);
    }
}

DMA::DMA(
    const std::string &name,
    s_ptr<Block> parent,
    int line_size)
    : Logic(name, parent), line_size(line_size) {
    ports["0"] = em()->CreatePort("0", this);
    ports["1"] = em()->CreatePort("1", this);
}


void DRAMSimPowerCallback(double a, double b, double c, double d)
{
    // CDCOUT("DRAMSimPowerCallback" << std::endl, dramDebugLevel);
}

DDR::DDR(
    const std::string &name, s_ptr<Block> parent,
    int num_ports, int line_size, int size)
    : Memory(name, parent, num_ports, line_size, size) {

	read_cb = new DRAMSim::Callback<DDR, void, unsigned, uint64_t, uint64_t>(this, &DDR::ReadCompleteCallback);
	write_cb = new DRAMSim::Callback<DDR, void, unsigned, uint64_t, uint64_t>(this, &DDR::WriteCompleteCallback);

    mem = DRAMSim::getMemorySystemInstance(
        "ini/DDR2_micron_16M_8b_x8_sg3E.ini", "system.ini", "../DRAMSim2", "example_app", 16384);
	mem->RegisterCallbacks(read_cb, write_cb, nullptr);
    transactionSize = TRANSACTION_SIZE;

    CDCOUT("created MemorySystemInstance" << std::endl, dramDebugLevel);

    //uint64_t addr = 0x900012;
    //ExampleTransactions(addr);
}

bool DDR::ProcessEvent(Event event) {
    CDCOUT("DDR: Processing event at clock: " << event.clock << std::endl, dramDebugLevel);

    if(event.io_direction != IODirection::In) {
        return false;  // Don't handle outgoing traffic from DDR just yet
    }

    if(event.is_write) {
        CDCOUT("DDR: write: ", dramDebugLevel);
    	mem->addTransaction(true, event.address);
    } else {
        CDCOUT("DDR: read: ", dramDebugLevel);
    	mem->addTransaction(false, event.address);
    }
    CDCOUT("clock: " << event.clock
        << " addr: " << event.address
        << " size: " << event.data_size << std::endl, dramDebugLevel);
    return true;
}

void DDR::ClockTick(Clock clock) {
    mem->update();
}

void DDR::ReadCompleteCallback(unsigned id, uint64_t address, uint64_t clock_cycle)
{
    CDCOUT("ReadCompleteCallback" << std::endl, dramDebugLevel);
    em()->QueueEvent(Event(
        clock_cycle, ports["0"]->port_id, IODirection::Out,
        address, 0, false
    ));
}

void DDR::WriteCompleteCallback(unsigned id, uint64_t address, uint64_t clock_cycle)
{
    CDCOUT("WriteCompleteCallback" << std::endl, dramDebugLevel);
    em()->QueueEvent(Event(
        clock_cycle, ports["0"]->port_id, IODirection::Out,
        address, 0, true
    ));
}

int DDR::ExampleTransactions(uint64_t addr)
{
	/* create a transaction and add it */
	bool isWrite = false;
	mem->addTransaction(isWrite, addr);

	// send a read to channel 1 on the same cycle
	addr = 1LL<<33 | addr;
	mem->addTransaction(isWrite, addr);

	for (int i=0; i<50; i++)
	{
		mem->update();
	}

	/* add another some time in the future */

	// send a write to channel 0
	addr = 0x900012;
	isWrite = true;
	mem->addTransaction(isWrite, addr);


	/* do a bunch of updates (i.e. clocks) -- at some point the callback will fire */
	for (int i=0; i<450; i++)
	{
		mem->update();
	}

	/* get a nice summary of this epoch */
    CDCOUT("ExampleTransactions: done with transactions" << std::endl, dramDebugLevel);
	mem->printStats(true);

	return 0;
}
