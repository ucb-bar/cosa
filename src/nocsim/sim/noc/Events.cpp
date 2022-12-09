#include <Events.h>

Port::Port(PortId port_id, const std::string& name, Block* parent)
    : port_id(port_id), name(name), parent(parent), twin() {

}

void ConnectPorts(s_ptr<Block> a, const std::string& aPort, s_ptr<Block> b, const std::string& bPort) {
    CDCOUT("Connecting " << a->name << " (" << aPort
        << ") with " << b->name << "(" << bPort << ")" << std::endl, dramDebugLevel);
    if(a->ports.find(aPort) == a->ports.end()) {
        CDCOUT("Could not find port (" << a->name << "): " << aPort << std::endl, dramDebugLevel);
        return;
    }
    if(b->ports.find(bPort) == b->ports.end()) {
        CDCOUT("Could not find port(" << b->name << "): " << bPort << std::endl, dramDebugLevel);
        return;
    }

    a->ports[aPort]->twin = b->ports[bPort];
    b->ports[bPort]->twin = a->ports[aPort];
}

Event::Event()
    : is_valid(false) {
}

Event::Event(Clock clock, PortId port_id, IODirection io_direction,
             Address address, DataSize data_size, bool is_write)
    : is_valid(true), clock(clock), port_id(port_id), io_direction(io_direction),
      address(address), data_size(data_size), is_write(is_write) {
}

bool operator<(const Event& lhs, const Event& rhs) {
    return lhs.clock < rhs.clock;
}


Block::Block(const std::string &name, s_ptr<Block> parent)
    : name(name), parent(parent) {
      CDCOUT("Constructing block: " << this->name << std::endl, dramDebugLevel);
      if(parent == nullptr)
          CDCOUT("parent == nullptr" << std::endl, dramDebugLevel);
}

s_ptr<EventManager> Block::em() {
    Block* block = this;
    while(block->parent != nullptr) {
        block = block->parent.get();
    }
    auto device = reinterpret_cast<Device*>(block);
    return device->_em;
}

bool Block::AddChild(const std::string &name, s_ptr<Block> block) {
    if(children.find(name) != children.end())
        return false;

    children[name] = block;
    return true;
}

s_ptr<Block> Block::GetChild(const std::string &name) {
    return children[name];
}

PortId Block::GetPortId(const std::string &name) {
    if(ports.find(name) == ports.end()) {
        CDCOUT("Could not find port (" << this->name << "): " << name << std::endl, dramDebugLevel);
        return -1;
    }
    return ports[name]->port_id;
}

bool Block::ProcessEvent(Event event) {
    CDCOUT("Block: Processing event at clock: " << event.clock << std::endl, dramDebugLevel);
    return false;
}

void Block::ClockTick(Clock clock) {

}

EventQueue::EventQueue(const std::string& name)
    : name(name) {

}

EventManager::EventManager()
    : avail_port_id(1), event_queue("Queued Events"), watched_ports(),
    watched_events("Watched Events"), blocks_to_tick(), curr_clock(0), ports1() {
    CDCOUT("Creating event manager" << std::endl, dramDebugLevel);
}

s_ptr<Port> EventManager::CreatePort(const std::string& name, Block* parent) {
    PortId port_id = avail_port_id;
    CDCOUT("Creating port: " << port_id
        << " for block: " << parent->name << std::endl, dramDebugLevel);
    auto p = std::make_shared<Port>(port_id, name, parent);
    this->ports1[port_id] = p;
    avail_port_id++;

    return p;
}

bool EventManager::QueueEvent(Event event) {
    event_queue.Push(event);
    return true;
}

void EventManager::RunToNextEvent() {
    if(event_queue.events.empty()) {
        return;
    }

    Event event;
    event_queue.Pop(event);

    if(watched_ports.find(event.port_id) != watched_ports.end()){
        watched_events.Push(event);
    }
}

void EventManager::ClockTick(Clock clock) {
    CDCOUT("clock: " << clock << std::endl, dramDebugLevel);
    curr_clock = clock;

    // Tick all the blocks so that you can process any events
    // they generated this clock
    for(auto iter = blocks_to_tick.begin(); iter != blocks_to_tick.end(); iter++) {
        (*iter)->ClockTick(curr_clock);
    }

    if(event_queue.HasEvent(curr_clock)) {
        Event event;
        event_queue.Pop(event);

        if(ports1.find(event.port_id) == ports1.end()) {
            CDCOUT("Cannot find port_id: " << event.port_id << std::endl, dramDebugLevel);
            return;
        }

        if(event.io_direction == IODirection::In) {
            auto port = ports1[event.port_id];
            auto block = port->parent;
            CDCOUT("Found block: " << block->name
                << " port: " << port->name
                << " port_id: " << event.port_id << std::endl, dramDebugLevel);
            block->ProcessEvent(event);
        }

        if(watched_ports.find(event.port_id) != watched_ports.end()
            && event.io_direction == IODirection::Out){
            watched_events.Push(event);
        }
    }

}

bool EventManager::WatchPort(PortId port_id) {
    watched_ports.insert(port_id);
    return true;
}

void EventManager::RegisterForClockTick(s_ptr<Block> block) {
    blocks_to_tick.push_back(block);
}


Event EventManager::GetWatchedEvent() {
    Event e;
    if(watched_events.HasEvent(curr_clock)) {
        watched_events.Pop(e);
    }
    return e;
}

Device::Device(const std::string &name, s_ptr<EventManager> _em)
    : Block(name, nullptr), _em(_em) {
}

bool Device::start() {
    return true;
}
