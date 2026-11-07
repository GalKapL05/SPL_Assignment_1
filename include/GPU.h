#pragma once

#include "Request.h"
#include "MemoryBlock.h"

using std::move;
using std::vector;
using std::string;

struct PrefixMatch {
    size_t matchedTokens;
    vector<size_t> blockIds;
};

struct PrefixInfo {
    string modelId;
    vector<string> prefixKey;
};

struct Reservation {
    size_t blocks;
    int requestId;
};

class GPU {
public:
    GPU(size_t blocksNum, size_t blockSize);
    
    ~GPU();
    GPU(const GPU& other);
    GPU& operator=(const GPU& other);
    GPU(GPU&& other);
    GPU& operator=(GPU&& other);

    size_t availableSlots() const;
    size_t getBlocksForMemory(size_t memory) const;

    void reserve(size_t n, const Request& request);
    void unreserve(size_t n, const Request& request);

    size_t allocateBlock(const Request& request);

    void addMemoryBlock(const GenerativeRequest& request, size_t slot);
    void addMemoryBlock(const EmbeddingRequest& request, size_t slot);

    void acquire(size_t slot);
    void release(size_t slot);

    void copyMemoryFromOtherGPU(const GPU& other, const vector<size_t>& blocks);
    void assignPrefixToBlock(size_t block, const vector<string>& prefixTokens, const string& modelId);

    int readFromBlock(size_t blockId, size_t offset) const;
    void writeToBlock(size_t blockId, size_t offset, int value);

    PrefixMatch findLongestCachedPrefix(const string& modelId, size_t embeddingSize, const vector<string>& promptTokens) const;

    const size_t& getBlockSize() const;
    const size_t& getCacheHits() const;
    const size_t& getEvictions() const;

private:
    size_t blocksNum;
    size_t blockSize;
    vector<MemoryBlock*> memory;
    vector<PrefixInfo> prefixesInfo;
    
    size_t totalReserved;
    vector<Reservation> requestReservations;
    
    size_t cacheHits;
    size_t evictions;

    void clearMemory();
};