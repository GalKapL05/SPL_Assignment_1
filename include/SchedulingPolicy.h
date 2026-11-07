#pragma once

#include "GPU.h"

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    virtual vector<int> choose(const vector<const Request*>& eligible, const vector<GPU>& gpus) const = 0;
    virtual SchedulingPolicy* clone() const = 0;
};

class ArrivalOrderPolicy : public SchedulingPolicy { 
    virtual ~ArrivalOrderPolicy() = default;
    vector<int> choose(const vector<const Request*>& eligible, const vector<GPU>& gpus) const override;
    SchedulingPolicy* clone() const override;
};
class ShortestJobPolicy : public SchedulingPolicy { 
    virtual ~ShortestJobPolicy() = default;
    vector<int> choose(const vector<const Request*>& eligible, const vector<GPU>& gpus) const override;
    SchedulingPolicy* clone() const override;
};
class LongestPrefixPolicy : public SchedulingPolicy {
    virtual ~LongestPrefixPolicy() = default;
    vector<int> choose(const vector<const Request*>& eligible, const vector<GPU>& gpus) const override;
    SchedulingPolicy* clone() const override;
};