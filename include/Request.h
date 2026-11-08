#pragma once

#include <vector>
#include <string>

#include "GPU.h"
#include "Model.h"

using std::vector;
using std::string;

class GPU;
class Model;

class Request{
protected:
    int id;
    const Model& model;
    int admissionCycle;
    vector<string> tokens;
    vector<size_t> memoryBlocks;
    size_t reservedBlocks;
public:
    Request(int id, const Model& model, int admissionCycle, string prompt);
    virtual ~Request() = default;

    virtual size_t getNeededMemory(const GPU& gpu) const = 0;
    virtual size_t getMaxNeededMemory() const = 0;
    virtual vector<string> getCacheableTokens() const = 0;
    virtual void allocateOneBlock(GPU& gpu, size_t slot) const = 0;
    virtual Request* clone(const Model& model) const = 0;
    void deallocateMemory(GPU& gpu);
    virtual void act(GPU& gpu) = 0;
    virtual bool finished() const = 0;
    virtual void reserveMemory(GPU& gpu) = 0;
    virtual string getOutput() const = 0;
    virtual size_t getNeedToUnreserve() const = 0;
    
    int getId() const;
    const Model& getModel() const;
    int getAdmissionCycle() const;
    const vector<string>& getTokens() const;
};

class GenerativeRequest: public Request {
private:
    string scriptedOutput;
    size_t maxOutputTokens;
    vector<string> outputTokens;
    size_t generatedTokensCount;
public:
    GenerativeRequest(int id, const Model& model, int admissionCycle, string prompt, string scriptedOutput, size_t maxOutputTokens);
    virtual ~GenerativeRequest() = default;

    size_t getNeededMemory(const GPU& gpu) const override;
    size_t getMaxNeededMemory() const override;
    vector<string> getCacheableTokens() const override;
    void allocateOneBlock(GPU& gpu, size_t slot) const override;
    Request* clone(const Model& model) const override;
    void act(GPU& gpu) override;
    bool finished() const override;
    void reserveMemory(GPU& gpu) override;
    string getOutput() const override;
    size_t getNeedToUnreserve() const override;
};

class EmbeddingRequest: public Request {
private:
    bool isFinished;
    size_t processedTokens;
    vector<int> accumulatedSum;
public:
    EmbeddingRequest(int id, const Model& model, int admissionCycle, string prompt);
    virtual ~EmbeddingRequest() = default;

    size_t getNeededMemory(const GPU& gpu) const override;
    size_t getMaxNeededMemory() const override;
    vector<string> getCacheableTokens() const override;
    void allocateOneBlock(GPU& gpu, size_t slot) const override;
    Request* clone(const Model& model) const override;
    void act(GPU& gpu) override;
    bool finished() const override;
    void reserveMemory(GPU& gpu) override;
    string getOutput() const override;
    size_t getNeedToUnreserve() const override;
};