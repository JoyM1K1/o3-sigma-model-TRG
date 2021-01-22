#ifndef O3_SIGMA_MODEL_TIME_COUNTER_HPP
#define O3_SIGMA_MODEL_TIME_COUNTER_HPP

#include <string>
#include <chrono>

class time_counter {
private:
    std::chrono::system_clock::time_point _start;
    std::chrono::system_clock::time_point _end;
public:
    time_counter();

    void start();

    void end();

    std::string duration_cast_to_string();
};

#endif //O3_SIGMA_MODEL_TIME_COUNTER_HPP
