#include "MemoryBlock.h"

MemoryBlock::MemoryBlock(size_t size)
    : size(size), refCount(0), data(new int[size]()) {}

MemoryBlock::~MemoryBlock() {
    delete[] data;
    data = nullptr;
}

MemoryBlock::MemoryBlock(const MemoryBlock& other)
    : size(other.size), refCount(0), data(new int[other.size]) {
    copy(other.data, other.data + other.size, data);
}

MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    if (this != &other) {
        delete[] data;
        data = new int[other.size];
        copy(other.data, other.data + other.size, data);
        size = other.size;
        refCount = 0;
    }
    return *this;
}

MemoryBlock::MemoryBlock(MemoryBlock&& other) :
    size(other.size), refCount(other.refCount), data(other.data) {
    other.data = nullptr;
    other.size = 0;
    other.refCount = 0;
}

MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) {
    if (this != &other) {
        delete[] data;

        data = other.data;
        size = other.size;
        refCount = other.refCount;

        other.data = nullptr;
        other.size = 0;
        other.refCount = 0;
    }
    return *this;
}

int MemoryBlock::getRefCount() const {
    return refCount;
}

const size_t& MemoryBlock::getSize() const {
    return size;
}

void MemoryBlock::writeToBlock(size_t offset, int value) {
    data[offset] = value;
}

int MemoryBlock::readFromBlock(size_t offset) const {
    return data[offset];
}

void MemoryBlock::acquire() {
    refCount++;
}

IdleAction MemoryBlock::release() {
    if (refCount > 0) {
        --refCount;
    }
    if (refCount == 0) {
        return onLastRelease();
    }
    return IdleAction::KeepCached;
}

KVBlock::KVBlock(size_t size) : MemoryBlock(size) {}

IdleAction KVBlock::onLastRelease() const {
    return IdleAction::KeepCached;
}

MemoryBlock* KVBlock::clone() const {
    return new KVBlock(*this);
}

EmbeddingWorkspace::EmbeddingWorkspace(size_t size) : MemoryBlock(size) {}

IdleAction EmbeddingWorkspace::onLastRelease() const {
    return IdleAction::Destroy;
}

MemoryBlock* EmbeddingWorkspace::clone() const {
    return new EmbeddingWorkspace(*this);
}