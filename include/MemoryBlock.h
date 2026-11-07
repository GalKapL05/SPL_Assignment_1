#pragma once

#include <string>

using std::copy;

enum class IdleAction { KeepCached, Destroy };

class MemoryBlock{
protected:
    size_t size;
    int refCount;
    int* data;
public:
    MemoryBlock(size_t size);
    
    // Rule of 5
    virtual ~MemoryBlock();                             // 1. Destructor
    MemoryBlock(const MemoryBlock& other);              // 2. Copy Constructor
    MemoryBlock& operator=(const MemoryBlock& other);   // 3. Copy Assignment Operator
    MemoryBlock(MemoryBlock&& other);                   // 4. Move Constructor
    MemoryBlock& operator=(MemoryBlock&& other);        // 5. Move Assignment Operator

    int readFromBlock(size_t offset) const;
    void writeToBlock(size_t offset, int value);

    void acquire();
    IdleAction release();

    virtual IdleAction onLastRelease() const = 0;
    virtual MemoryBlock* clone() const = 0;
};

class KVBlock: public MemoryBlock{
public:
    KVBlock(size_t size);

    virtual IdleAction onLastRelease() const;
    virtual MemoryBlock* clone() const;
};

class EmbeddingWorkspace: public MemoryBlock{
public:
    EmbeddingWorkspace(size_t size);

    virtual IdleAction onLastRelease() const;
    virtual MemoryBlock* clone() const;
};