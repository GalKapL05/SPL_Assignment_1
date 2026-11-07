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
    
    virtual ~MemoryBlock();                             
    MemoryBlock(const MemoryBlock& other);              
    MemoryBlock& operator=(const MemoryBlock& other);   
    MemoryBlock(MemoryBlock&& other);                   
    MemoryBlock& operator=(MemoryBlock&& other);     

    int getRefCount() const;
    const size_t& getSize() const;

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