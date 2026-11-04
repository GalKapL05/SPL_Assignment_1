#ifndef SCHEDULING_POLICY_H_
#define SCHEDULING_POLICY_H

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    virtual std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const = 0;
    virtual SchedulingPolicy* clone() const = 0;
};

class ArrivalOrderPolicy : public SchedulingPolicy { 

};
class ShortestJobPolicy : public SchedulingPolicy { 

};
class LongestPrefixPolicy : public SchedulingPolicy {

};


#endif