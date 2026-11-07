#include "SchedulingPolicy.h"

vector<int> ArrivalOrderPolicy::choose(const vector<const Request*>& eligible, const vector<GPU>& gpus) const {
    (void)gpus;
    std::vector<int> order;
    order.reserve(eligible.size());

    for (size_t i = 0; i < eligible.size(); i++) {
        order.push_back(static_cast<int>(i));
    }

    return order;
}

SchedulingPolicy* ArrivalOrderPolicy::clone() const {
    return new ArrivalOrderPolicy(*this);
}

vector<int> ShortestJobPolicy::choose(const vector<const Request*>& eligible, const vector<GPU>& gpus) const {
    std::vector<int> order;
    order.reserve(eligible.size());

    if (eligible.empty()) {
        return order;
    }

    vector<size_t> requiredBlocks(eligible.size(), 0);
    for (size_t i = 0; i < eligible.size(); i++) {
        order.push_back(static_cast<int>(i));
        
        if (!gpus.empty()) {
            requiredBlocks[i] = gpus[0].getBlocksForMemory(eligible[i]->getMaxNeededMemory());
        }
    }

    for (size_t i = 0; i < order.size(); i++) {
        for (size_t j = 0; j + 1 < order.size() - i; j++) {
            if (requiredBlocks[order[j]] > requiredBlocks[order[j + 1]]) {
                int temp = order[j];
                order[j] = order[j + 1];
                order[j + 1] = temp;
            }
        }
    }

    return order;
}

SchedulingPolicy* ShortestJobPolicy::clone() const {
    return new ShortestJobPolicy(*this);
}

vector<int> LongestPrefixPolicy::choose(const vector<const Request*>& eligible, const vector<GPU>& gpus) const {
    vector<int> order;
    order.reserve(eligible.size());

    vector<size_t> maxPrefixes(eligible.size(), 0);

    for (size_t i = 0; i < eligible.size(); i++) {
        order.push_back(static_cast<int>(i));
        const Request* req = eligible[i];
        size_t bestMatch = 0;

        for (size_t j = 0; j < gpus.size(); j++) {
            PrefixMatch match = gpus[j].findLongestCachedPrefix(
                req->getModel().getModelId(),
                req->getModel().getEmbeddingSize(),
                req->getTokens()
            );
            if (match.matchedTokens > bestMatch) {
                bestMatch = match.matchedTokens;
            }
        }
        maxPrefixes[i] = bestMatch;
    }

    for (size_t i = 0; i < order.size(); i++) {
        for (size_t j = 0; j + 1 < order.size() - i; j++) {
            if (maxPrefixes[order[j]] < maxPrefixes[order[j + 1]]) {
                int temp = order[j];
                order[j] = order[j + 1];
                order[j + 1] = temp;
            }
        }
    }

    return order;
}

SchedulingPolicy* LongestPrefixPolicy::clone() const {
    return new LongestPrefixPolicy(*this);
}