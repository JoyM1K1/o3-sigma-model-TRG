#include "../include/time_counter.hpp"
#include <sstream>
#include <iomanip>

time_counter::time_counter() {
    _start = std::chrono::system_clock::now();
    _end = std::chrono::system_clock::now();
}

void time_counter::start() {
    _start = std::chrono::system_clock::now();
}

void time_counter::end() {
    _end = std::chrono::system_clock::now();
}

std::string time_counter::duration_cast_to_string() {
    long long hours = std::chrono::duration_cast<std::chrono::hours>(_end - _start).count();
    long long minutes = std::chrono::duration_cast<std::chrono::minutes>(_end - _start).count() % 60;
    long long seconds = std::chrono::duration_cast<std::chrono::seconds>(_end - _start).count() % 60;
    long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(_end - _start).count() % 1000;
    std::stringstream res;
    if (hours > 0) {
        res << std::setw(2) << hours << " h " << std::setw(2) << minutes << " m " << std::setw(2) << seconds << " s " << std::setw(3) << milli << " ms";
    } else if (minutes > 0) {
        res << std::setw(2) << minutes << " m " << std::setw(2) << seconds << " s " << std::setw(3) << milli << " ms";
    } else if (seconds > 0) {
        res << std::setw(2) << seconds << " s " << std::setw(3) << milli << " ms";
    } else {
        res << std::setw(3) << milli << " ms";
    }
    return res.str();
}