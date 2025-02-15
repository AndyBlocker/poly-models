#ifndef __TIME_UTILS_HPP__
#define __TIME_UTILS_HPP__

#include <chrono>
#include <vector>
#include <string>

enum class OpType {
    IM2COL = 0,
    MATMUL,
    POOL,
    OTHERS,
    OVERALL,

    ENDOP, // add all new op before this, dont change this
};

class GlobalProfiler {
public:
    static GlobalProfiler& instance() {
        static GlobalProfiler profiler;
        return profiler;
    }

    void add_time(OpType op, double ms) {
        times_[static_cast<int>(op)] += ms;
    }

    double get_time(OpType op) const {
        return times_[static_cast<int>(op)];
    }

    void reset() {
        for(auto &t : times_) {
            t = 0.0;
        }
    }

private:
    GlobalProfiler() {
        times_.resize(static_cast<int>(OpType::ENDOP), 0.0);
    }
    std::vector<double> times_;
};

class ScopedTimer {
public:
    explicit ScopedTimer(OpType op_type)
        : op_type_(op_type),
          start_(std::chrono::steady_clock::now())
    {}

    ~ScopedTimer() {
        auto end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start_).count();
        GlobalProfiler::instance().add_time(op_type_, ms);
    }

private:
    OpType op_type_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

#endif