// path-60 experiment: P60_SYNC_TRACE=<file> writes one line per synchronization event with its
// CLOCK_UPTIME_RAW time in ns (the clock of MoltenVK's frame profiler and of the verify hook).
#pragma once

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <time.h>

namespace P60Trace {
inline std::FILE* File() {
    static std::FILE* const file = [] {
        const char* path = std::getenv("P60_SYNC_TRACE");
        return path ? std::fopen(path, "w") : nullptr;
    }();
    return file;
}
inline bool On() {
    return File() != nullptr;
}
inline unsigned long long Now() {
#ifdef __APPLE__
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
    return 0;
#endif
}
inline void Event(const char* tag, const char* where = "-", unsigned long long a = 0,
                  unsigned long long b = 0) {
    if (std::FILE* const file = File()) {
        static std::mutex mutex;
        const unsigned long long now = Now();
        std::scoped_lock lock{mutex};
        std::fprintf(file, "%llu %s %s %llu %llu\n", now, tag, where, a, b);
    }
}
} // namespace P60Trace
