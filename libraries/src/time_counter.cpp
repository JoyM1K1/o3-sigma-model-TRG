#include "../include/time_counter.hpp"

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
    std::string res;
    if (hours > 0) {
        res = std::to_string(hours) + " h " + std::to_string(minutes) + " m " + std::to_string(seconds) + " s " + std::to_string(milli) + " ms";
    } else if (minutes > 0) {
        res = std::to_string(minutes) + " m " + std::to_string(seconds) + " s " + std::to_string(milli) + " ms";
    } else if (seconds > 0) {
        res = std::to_string(seconds) + " s " + std::to_string(milli) + " ms";
    } else {
        res = std::to_string(milli) + " ms";
    }
    return res;
}