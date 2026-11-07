#include "Request.h"

Request::Request(int id, const Model& model, int admissionCycle, string prompt) : 
    id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)),
    memoryBlocks(), reservedBlocks(0) {}

void Request::deallocateMemory(GPU& gpu) {
    for (size_t i = 0; i < memoryBlocks.size(); i++) {
        gpu.release(memoryBlocks[i]);
    }
    memoryBlocks.clear();
}

int Request::getId() const {
    return id;
}

const Model& Request::getModel() const {
    return model;
}

int Request::getAdmissionCycle() const {
    return admissionCycle;
}

const vector<string>& Request::getTokens() const {
    return tokens;
}

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle,
                                     string prompt, string scriptedOutput,
                                     size_t maxOutputTokens) :
    Request(id, model, admissionCycle, prompt),
    scriptedOutput(scriptedOutput),
    maxOutputTokens(maxOutputTokens),
    outputTokens(model.tokenizer(scriptedOutput)),
    generatedTokensCount(0) {}

size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const {
    size_t currentTokens = tokens.size() + generatedTokensCount;
    size_t neededUnits = currentTokens * model.getEmbeddingSize();
    return gpu.getBlocksForMemory(neededUnits);
}

size_t GenerativeRequest::getMaxNeededMemory() const {
    size_t totalTokens = tokens.size() + maxOutputTokens;
    return totalTokens * model.getEmbeddingSize();
}

vector<string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}

void GenerativeRequest::allocateOneBlock(GPU& gpu, size_t slot) const {
    (void)gpu;
    (void)slot;
}

Request* GenerativeRequest::clone(const Model& modelRef) const {
    GenerativeRequest* copy = new GenerativeRequest(id, modelRef, admissionCycle, "", scriptedOutput, maxOutputTokens);
    copy->tokens = this->tokens;
    copy->outputTokens = this->outputTokens;
    copy->generatedTokensCount = this->generatedTokensCount;
    copy->memoryBlocks = this->memoryBlocks;
    copy->reservedBlocks = this->reservedBlocks;
    return copy;
}

void GenerativeRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }

    size_t tokensPerBlock = gpu.getBlockSize() / model.getEmbeddingSize();
    if (tokensPerBlock == 0) {
        return;
    }

    if (generatedTokensCount < outputTokens.size()) {
        size_t currentTokenIndex = tokens.size() + generatedTokensCount;
        size_t blockIdx = currentTokenIndex / tokensPerBlock;

        if (blockIdx < memoryBlocks.size()) {
            size_t slot = memoryBlocks[blockIdx];
            size_t offset = (currentTokenIndex % tokensPerBlock) * model.getEmbeddingSize();
            model.generateTokenEmbedding(outputTokens, gpu, slot, offset);
        }
    }

    ++generatedTokensCount;
}

bool GenerativeRequest::finished() const {
    return generatedTokensCount >= maxOutputTokens || generatedTokensCount >= outputTokens.size();
}

void GenerativeRequest::reserveMemory(GPU& gpu) {
    size_t neededBlocks = gpu.getBlocksForMemory(getMaxNeededMemory());
    if (neededBlocks > memoryBlocks.size()) {
        reservedBlocks = neededBlocks - memoryBlocks.size();
        gpu.reserve(reservedBlocks, *this);
    }
}

string GenerativeRequest::getOutput() const {
    string result = "";
    size_t limit = generatedTokensCount;
    if (limit > outputTokens.size()) {
        limit = outputTokens.size();
    }

    for (size_t i = 0; i < limit; i++) {
        if (i > 0) {
            result += " ";
        }
        result += outputTokens[i];
    }
    return result;
}

size_t GenerativeRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, string prompt)
    : Request(id, model, admissionCycle, prompt),
      isFinished(false) {}

size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const {
    size_t neededUnits = tokens.size() * model.getEmbeddingSize();
    return gpu.getBlocksForMemory(neededUnits);
}

size_t EmbeddingRequest::getMaxNeededMemory() const {
    return tokens.size() * model.getEmbeddingSize();
}

vector<string> EmbeddingRequest::getCacheableTokens() const {
    return tokens;
}

void EmbeddingRequest::allocateOneBlock(GPU& gpu, size_t slot) const {
    (void)gpu;
    (void)slot;
}

Request* EmbeddingRequest::clone(const Model& modelRef) const {
    EmbeddingRequest* copy = new EmbeddingRequest(id, modelRef, admissionCycle, "");
    copy->tokens = this->tokens;
    copy->isFinished = this->isFinished;
    copy->memoryBlocks = this->memoryBlocks;
    copy->reservedBlocks = this->reservedBlocks;
    return copy;
}

void EmbeddingRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }

    size_t tokensPerBlock = gpu.getBlockSize() / model.getEmbeddingSize();
    if (tokensPerBlock == 0) {
        return;
    }

    for (size_t i = 0; i < tokens.size(); i++) {
        size_t blockIdx = i / tokensPerBlock;
        if (blockIdx < memoryBlocks.size()) {
            size_t slot = memoryBlocks[blockIdx];
            size_t offset = (i % tokensPerBlock) * model.getEmbeddingSize();
            model.generateTokenEmbedding(tokens, gpu, slot, offset);
        }
    }

    isFinished = true;
}

bool EmbeddingRequest::finished() const {
    return isFinished;
}

void EmbeddingRequest::reserveMemory(GPU& gpu) {
    size_t neededBlocks = gpu.getBlocksForMemory(getMaxNeededMemory());
    if (neededBlocks > memoryBlocks.size()) {
        reservedBlocks = neededBlocks - memoryBlocks.size();
        gpu.reserve(reservedBlocks, *this);
    }
}

string EmbeddingRequest::getOutput() const {
    return "";
}

size_t EmbeddingRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}