#include "GPU.h"

GPU::GPU(size_t blocksNum, size_t blockSize) : 
    blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr),
    prefixesInfo(blocksNum, PrefixInfo{"", {}}), totalReserved(0),
    requestReservations(), cacheHits(0), evictions(0) {}

void GPU::clearMemory() {
    for (size_t i = 0; i < blocksNum; i++) {
        if (memory[i] != nullptr) {
            delete memory[i];
            memory[i] = nullptr;
        }
    }
}

GPU::~GPU() {
    clearMemory();
}

GPU::GPU(const GPU& other) : 
    blocksNum(other.blocksNum), blockSize(other.blockSize), memory(other.blocksNum, nullptr),
    prefixesInfo(other.prefixesInfo), totalReserved(other.totalReserved),
    requestReservations(other.requestReservations), cacheHits(other.cacheHits), evictions(other.evictions) {
    for (size_t i = 0; i < blocksNum; i++) {
        if (other.memory[i] != nullptr) {
            memory[i] = other.memory[i]->clone();
            for (int j = 0; j < other.memory[i]->getRefCount(); j++) {
                memory[i]->acquire();
            }
        }
    }
}

GPU& GPU::operator=(const GPU& other) {
    if (this != &other) {
        clearMemory();

        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        memory.assign(blocksNum, nullptr);
        prefixesInfo = other.prefixesInfo;
        totalReserved = other.totalReserved;
        requestReservations = other.requestReservations;
        cacheHits = other.cacheHits;
        evictions = other.evictions;

        for (size_t i = 0; i < blocksNum; i++) {
            if (other.memory[i] != nullptr) {
                memory[i] = other.memory[i]->clone();
                for (int j = 0; j < other.memory[i]->getRefCount(); j++) {
                    memory[i]->acquire();
                }
            }
        }
    }
    return *this;
}

GPU::GPU(GPU&& other) : 
    blocksNum(other.blocksNum), blockSize(other.blockSize), memory(move(other.memory)),
    prefixesInfo(move(other.prefixesInfo)), totalReserved(other.totalReserved),
    requestReservations(move(other.requestReservations)), cacheHits(other.cacheHits),
    evictions(other.evictions) {
    other.blocksNum = 0;
    other.blockSize = 0;
    other.totalReserved = 0;
    other.cacheHits = 0;
    other.evictions = 0;
}

GPU& GPU::operator=(GPU&& other) {
    if (this != &other) {
        clearMemory();

        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        memory = move(other.memory);
        prefixesInfo = move(other.prefixesInfo);
        totalReserved = other.totalReserved;
        requestReservations = move(other.requestReservations);
        cacheHits = other.cacheHits;
        evictions = other.evictions;

        other.blocksNum = 0;
        other.blockSize = 0;
        other.totalReserved = 0;
        other.cacheHits = 0;
        other.evictions = 0;
    }
    return *this;
}

size_t GPU::availableSlots() const {
    size_t activeCount = 0;
    for (size_t i = 0; i < blocksNum; i++) {
        if (memory[i] != nullptr && memory[i]->getRefCount() > 0) {
            ++activeCount;
        }
    }
    if (activeCount + totalReserved >= blocksNum) {
        return 0;
    }
    return blocksNum - activeCount - totalReserved;
}

size_t GPU::getBlocksForMemory(size_t memory) const {
    if (memory == 0) {
        return 0;
    }
    return (memory + blockSize - 1) / blockSize;
}

void GPU::reserve(size_t n, const Request& request) {
    totalReserved += n;
    requestReservations[request.getId()].blocks += n;
}

void GPU::unreserve(size_t n, const Request& request) {
    totalReserved = (totalReserved >= n) ? (totalReserved - n) : 0;
    for (size_t i = 0; i < requestReservations.size(); i++) {
        if (requestReservations[i].requestId == request.getId()) {
            requestReservations[i].blocks = (requestReservations[i].blocks >= n) ? (requestReservations[i].blocks - n) : 0;
        }
    }
}

size_t GPU::allocateBlock(const Request& request) {
    int targetSlot = -1;

    for (size_t i = 0; i < blocksNum; i++) {
        if (memory[i] == nullptr) {
            targetSlot = static_cast<int>(i);
            break;
        }
    }

    if (targetSlot == -1) {
        for (size_t i = 0; i < blocksNum; i++) {
            if (memory[i] != nullptr && memory[i]->getRefCount() == 0) {
                targetSlot = static_cast<int>(i);
                delete memory[targetSlot];
                memory[targetSlot] = nullptr;
                prefixesInfo[targetSlot].modelId = "";
                prefixesInfo[targetSlot].prefixKey.clear();
                ++evictions;
                break;
            }
        }
    }

    size_t chosenSlot = static_cast<size_t>(targetSlot);
    request.allocateOneBlock(*this, chosenSlot);
    unreserve(1, request);
    acquire(chosenSlot);
    return chosenSlot;
}

void GPU::addMemoryBlock(const GenerativeRequest& request, size_t slot) {
    memory[slot] = new KVBlock(blockSize);
    prefixesInfo[slot].modelId = "";
    prefixesInfo[slot].prefixKey.clear();
}

void GPU::addMemoryBlock(const EmbeddingRequest& request, size_t slot) {
    memory[slot] = new EmbeddingWorkspace(blockSize);
    prefixesInfo[slot].modelId = "";
    prefixesInfo[slot].prefixKey.clear();
}

void GPU::assignPrefixToBlock(size_t block, const vector<string>& prefixTokens, const string& modelId) {
    for (size_t i = 0; i < blocksNum; i++) {
        if (i != block && memory[i] != nullptr && prefixesInfo[i].modelId == modelId) {
            if (prefixesInfo[i].prefixKey == prefixTokens) {
                return;
            }
        }
    }
    prefixesInfo[block].modelId = modelId;
    prefixesInfo[block].prefixKey = prefixTokens;
}

PrefixMatch GPU::findLongestCachedPrefix(const string& modelId, size_t embeddingSize, const vector<string>& promptTokens) const {
    PrefixMatch match;
    match.matchedTokens = 0;
    if (embeddingSize == 0) {
        return match;
    }

    size_t tokensPerBlock = blockSize / embeddingSize;
    if (tokensPerBlock == 0) {
        return match;
    }

    size_t currentCovered = 0;
    size_t totalTokens = promptTokens.size();
    while (currentCovered + tokensPerBlock <= totalTokens) {
        int foundSlot = -1;
        vector<string> targetPrefix(promptTokens.begin(), promptTokens.begin() + currentCovered + tokensPerBlock);
        for (size_t i = 0; i < blocksNum; i++) {
            if (memory[i] != nullptr && prefixesInfo[i].modelId == modelId && prefixesInfo[i].prefixKey == targetPrefix) {
                foundSlot = static_cast<int>(i);
                break;
            }
        }

        if (foundSlot == -1) {
            break;
        }

        match.blockIds.push_back(static_cast<size_t>(foundSlot));
        currentCovered += tokensPerBlock;
    }

    match.matchedTokens = currentCovered;
    return match;
}

void GPU::copyMemoryFromOtherGPU(const GPU& other, const vector<size_t>& blocks) {
    vector<size_t> protectedSlots;

    for (size_t sourceSlot : blocks) {
        const string& modelId = other.prefixesInfo[sourceSlot].modelId;
        const vector<string>& key = other.prefixesInfo[sourceSlot].prefixKey;

        bool alreadyExists = false;
        for (size_t i = 0; i < blocksNum; i++) {
            if (memory[i] != nullptr && prefixesInfo[i].modelId == modelId && prefixesInfo[i].prefixKey == key) {
                alreadyExists = true;
                break;
            }
        }
        if (alreadyExists) {
            continue;
        }

        int targetSlot = -1;
        for (size_t i = 0; i < blocksNum; i++) {
            if (memory[i] == nullptr) {
                targetSlot = static_cast<int>(i);
                break;
            }
        }
        if (targetSlot == -1) {
            for (size_t i = 0; i < blocksNum; ++i) {
                if (memory[i] != nullptr && memory[i]->getRefCount() == 0) {
                    bool found = false;
                    for (size_t j = 0; j < protectedSlots.size(); j++) {
                        if (protectedSlots[j] == i) {
                            found = true;
                            break;
                        }
                    }

                    if (!found)
                    {
                        targetSlot = static_cast<int>(i);
                        delete memory[targetSlot];
                        memory[targetSlot] = nullptr;
                        prefixesInfo[targetSlot].modelId = "";
                        prefixesInfo[targetSlot].prefixKey.clear();
                        ++evictions;
                        break;
                    }
                }
            }
        }

        size_t destSlot = static_cast<size_t>(targetSlot);
        memory[destSlot] = other.memory[sourceSlot]->clone();
        prefixesInfo[destSlot].modelId = modelId;
        prefixesInfo[destSlot].prefixKey = key;
        protectedSlots.push_back(destSlot);
    }
}

void GPU::acquire(size_t slot) {
    if (memory[slot] != nullptr) {
        memory[slot]->acquire();
    }
}

void GPU::release(size_t slot) {
    if (memory[slot] == nullptr) {
        return;
    }

    IdleAction action = memory[slot]->release();
    if (memory[slot]->getRefCount() == 0) {
        if (action == IdleAction::Destroy || prefixesInfo[slot].prefixKey.empty()) {
            delete memory[slot];
            memory[slot] = nullptr;
            prefixesInfo[slot].modelId = "";
            prefixesInfo[slot].prefixKey.clear();
        }
    }
}

int GPU::readFromBlock(size_t blockId, size_t offset) const {
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(size_t blockId, size_t offset, int value) {
    memory[blockId]->writeToBlock(offset, value);
}

const size_t &GPU::getBlockSize() const
{
    return blockSize;
}

const size_t &GPU::getCacheHits() const { return cacheHits; }
const size_t& GPU::getEvictions() const { return evictions; }