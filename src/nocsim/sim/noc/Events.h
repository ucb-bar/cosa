#ifndef _EVENTS_HPP_
#define _EVENTS_HPP_

#include <memory>
#include <set>
#include <queue>
#include <map>
#include <iostream>

// To enable debug print
#include <hls_globals.h> 
static const int dramDebugLevel = 4;
using namespace ::std;

template<typename T> using s_ptr = std::shared_ptr<T>;
using Clock = uint64_t;
using PortId = int;
using Address = uint64_t;
using DataSize = size_t;

class Block;
class Event;
class EventManager;

struct Port {
    PortId port_id;
    const std::string name;
    Block* parent;
    s_ptr<Port> twin;

    Port(PortId port_id, const std::string& name, Block* parent);
};

class Block {
public:
    const std::string name;
    s_ptr<Block> parent;
    std::map<const std::string, s_ptr<Block>> children;
    std::map<const std::string, s_ptr<Port>> ports;

    s_ptr<EventManager> em();
    Block(const std::string &name, s_ptr<Block> parent = nullptr);
    bool AddChild(const std::string &name, s_ptr<Block> block);
    s_ptr<Block> GetChild(const std::string &name);
    PortId GetPortId(const std::string &name);
    virtual bool ProcessEvent(Event event);
    virtual void ClockTick(Clock clock);
};

void ConnectPorts(s_ptr<Block> a, const std::string& aPort, s_ptr<Block> b, const std::string& bPort);

enum IODirection {
    In = 0,
    Out = 1
};

struct Event {
    bool is_valid;
    Clock clock;
    PortId port_id;
    IODirection io_direction;
    Address address;
    DataSize data_size;
    bool is_write;

    Event();
    Event(Clock clock, PortId port_id, IODirection io_direction,
          Address address, DataSize data_size, bool is_write);
};

bool operator<(const Event& lhs, const Event& rhs);

struct EventQueue {
    const std::string name;
    std::priority_queue<Event> events;

    EventQueue(const std::string& name);

    void Push(Event e) {
        CDCOUT(name << ": pushing event" << std::endl, dramDebugLevel);
        events.push(e);
    }

    bool Pop(Event &e) {
        if(events.empty()) {
            return false;
        }
        e = events.top();
        CDCOUT(name << ": popping event: clock: " << e.clock
            << " port_id: " << e.port_id << std::endl, dramDebugLevel);

        events.pop();
        return true;
    }

    bool HasEvent(Clock clock) {
        if(events.empty()) {
            return false;
        }
        auto e = events.top();
        return (e.clock <= clock);
    }
};

class EventManager {
    PortId avail_port_id;
    EventQueue event_queue;
    std::set<PortId> watched_ports;
    EventQueue watched_events;
    std::vector<s_ptr<Block>> blocks_to_tick;

public:
    Clock curr_clock;
    std::map<PortId, s_ptr<Port>> ports1;

    EventManager();

    s_ptr<Port> CreatePort(const std::string& name, Block* parent);
    bool QueueEvent(Event event);
    void RunToNextEvent();
    void ClockTick(Clock clock);
    bool WatchPort(PortId portId);
    void RegisterForClockTick(s_ptr<Block> block);
    Event GetWatchedEvent();
};

class Device: public Block {
public:
    s_ptr<EventManager> _em;
    Device(const std::string &name, s_ptr<EventManager> em);

    bool start();
};


#endif
