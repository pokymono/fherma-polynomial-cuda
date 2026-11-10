#include "../candidates/ordered-streaming/cpu_affinity.h"
#include <cassert>
#include <iostream>
#include <thread>

int main() {
#if defined(__linux__)
    cpu_set_t before, after;
    assert(sched_getaffinity(0, sizeof(before), &before) == 0);
    const auto& placement = CpuPlacement::get();
    {
        CallerPlacement temporary;
        assert(sched_getaffinity(0, sizeof(after), &after) == 0);
        if (placement.caller() >= 0) {
            assert(CPU_COUNT(&after) == 1);
            assert(CPU_ISSET(placement.caller(), &after));
        }
    }
    assert(sched_getaffinity(0, sizeof(after), &after) == 0);
    assert(CPU_EQUAL(&before, &after));
    for (unsigned requested = 1; requested <= 16; ++requested) {
        unsigned lanes = placement.copy_lanes(requested);
        assert(lanes >= 1 && lanes <= requested);
        if (placement.physical.size() > 1)
            assert(lanes < placement.physical.size());
    }
    const unsigned copies = placement.copy_lanes(HOST_COPY_LANES);
    if (placement.physical.size() > copies) {
        for (unsigned lane = 0; lane < copies; ++lane)
            assert(placement.copy(lane) != placement.caller());
    }
    if (placement.physical.size() > copies + 1) {
        assert(placement.allocator() != placement.caller());
        for (unsigned lane = 0; lane < copies; ++lane)
            assert(placement.copy(lane) != placement.allocator());
    }
    std::thread helper([&] {
        CpuPlacement::pin(placement.copy(0));
        cpu_set_t current;
        assert(sched_getaffinity(0, sizeof(current), &current) == 0);
        if (!placement.physical.empty()) {
            assert(CPU_COUNT(&current) == 1);
            assert(CPU_ISSET(placement.copy(0), &current));
        }
    });
    helper.join();
    assert(sched_getaffinity(0, sizeof(after), &after) == 0);
    assert(CPU_EQUAL(&before, &after));
    std::cout << "Affinity placement and caller restoration passed; physical cores: "
              << placement.physical.size() << '\n';
#endif
}
