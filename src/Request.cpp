#include "Request.h"

Request::Request(int id, const Model& model, int admissionCycle, string prompt) : 
    id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)),
    memoryBlocks(), reservedBlocks(0) {}

void Request::deallocateMemory(GPU& gpu) {
    for (size_t i = 0; i < memoryBlocks.size(); i++) {
        gpu.release(memoryBlocks[i]);
    }
    memoryBlocks.clear();
    if (reservedBlocks > 0) {
        gpu.unreserve(reservedBlocks, *this);
        reservedBlocks = 0;
    }
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
    size_t n = tokens.size();
    size_t m = maxOutputTokens;
    size_t h = 0;

    size_t tokensPerBlock = gpu.getBlockSize() / model.getEmbeddingSize();
    if (tokensPerBlock > 0) {
        PrefixMatch match = gpu.findLongestCachedPrefix(model.getModelId(), model.getEmbeddingSize(), tokens);
        h = match.matchedTokens;
    }

    size_t remainingTokens = (n >= h) ? (n - h + m) : m;
    return remainingTokens * model.getEmbeddingSize();
}

size_t GenerativeRequest::getMaxNeededMemory() const {
    size_t totalTokens = tokens.size() + maxOutputTokens;
    return totalTokens * model.getEmbeddingSize();
}

vector<string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}

void GenerativeRequest::allocateOneBlock(GPU& gpu, size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
    GenerativeRequest* self = const_cast<GenerativeRequest*>(this);
    self->memoryBlocks.push_back(slot);
    if (self->reservedBlocks > 0) {
        self->reservedBlocks--;
    }
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

bool GenerativeRequest::finished() const {
    return generatedTokensCount >= maxOutputTokens || generatedTokensCount >= outputTokens.size();
}

void GenerativeRequest::reserveMemory(GPU& gpu) {
    size_t tokensPerBlock = gpu.getBlockSize() / model.getEmbeddingSize();
    if (tokensPerBlock > 0) {
        PrefixMatch match = gpu.findLongestCachedPrefix(model.getModelId(), model.getEmbeddingSize(), tokens);
        for (size_t slot : match.blockIds) {
            gpu.acquire(slot);
            memoryBlocks.push_back(slot);
        }
    }

    size_t neededBlocks = gpu.getBlocksForMemory(getMaxNeededMemory());
    if (neededBlocks > memoryBlocks.size()) {
        reservedBlocks = neededBlocks - memoryBlocks.size();
        gpu.reserve(reservedBlocks, *this);
    }

    size_t promptBlocksNeeded = (tokens.size() + tokensPerBlock - 1) / tokensPerBlock;
    while (memoryBlocks.size() < promptBlocksNeeded) {
        gpu.allocateBlock(*this);
    }

    size_t covered = 0;
    while (covered + tokensPerBlock <= tokens.size()) {
        size_t blockIdx = covered / tokensPerBlock;
        vector<string> prefixKey(tokens.begin(), tokens.begin() + covered + tokensPerBlock);
        gpu.assignPrefixToBlock(memoryBlocks[blockIdx], prefixKey, model.getModelId());
        covered += tokensPerBlock;
    }
}

void GenerativeRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }

    size_t tokensPerBlock = gpu.getBlockSize() / model.getEmbeddingSize();
    if (tokensPerBlock == 0) {
        return;
    }

    size_t currentTokenIndex = tokens.size() + generatedTokensCount;
    size_t neededBlocksCount = (currentTokenIndex / tokensPerBlock) + 1;

    while (memoryBlocks.size() < neededBlocksCount) {
        gpu.allocateBlock(*this);
    }

    if (generatedTokensCount < outputTokens.size()) {
        size_t blockIdx = currentTokenIndex / tokensPerBlock;
        size_t slot = memoryBlocks[blockIdx];
        size_t offset = (currentTokenIndex % tokensPerBlock) * model.getEmbeddingSize();
        
        vector<string> fullSequence = tokens;
        for (size_t i = 0; i <= generatedTokensCount; i++) {
            fullSequence.push_back(outputTokens[i]);
        }

        model.generateTokenEmbedding(outputTokens, gpu, slot, offset);

        if ((currentTokenIndex % tokensPerBlock) == tokensPerBlock - 1) {
            gpu.assignPrefixToBlock(slot, fullSequence, model.getModelId());
        }
    }

    ++generatedTokensCount;
}

string GenerativeRequest::getOutput() const {
    size_t limit = min(generatedTokensCount, outputTokens.size());
    if (limit == 0) {
        return "";
    }

    string result = outputTokens[0];
    for (size_t i = 1; i < limit; i++) {
        result += " " + outputTokens[i];
    }
    return result;
}

size_t GenerativeRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, string prompt)
    : Request(id, model, admissionCycle, prompt),
      isFinished(false),
      processedTokens(0),
      accumulatedSum(model.getEmbeddingSize(), 0) {}

size_t EmbeddingRequest::getNeededMemory(const GPU& gpu) const {
    (void)gpu;
    return 2 * model.getEmbeddingSize();
}

size_t EmbeddingRequest::getMaxNeededMemory() const {
    return 2 * model.getEmbeddingSize();
}

vector<string> EmbeddingRequest::getCacheableTokens() const {
    return vector<string>();
}

void EmbeddingRequest::allocateOneBlock(GPU& gpu, size_t slot) const {
    gpu.addMemoryBlock(*this, slot);
    EmbeddingRequest* self = const_cast<EmbeddingRequest*>(this);
    self->memoryBlocks.push_back(slot);
    if (self->reservedBlocks > 0) {
        self->reservedBlocks--;
    }
}

Request* EmbeddingRequest::clone(const Model& modelRef) const {
    EmbeddingRequest* copy = new EmbeddingRequest(id, modelRef, admissionCycle, "");
    copy->tokens = this->tokens;
    copy->isFinished = this->isFinished;
    copy->processedTokens = this->processedTokens;
    copy->accumulatedSum = this->accumulatedSum;
    copy->memoryBlocks = this->memoryBlocks;
    copy->reservedBlocks = this->reservedBlocks;
    return copy;
}

void EmbeddingRequest::reserveMemory(GPU& gpu) {
    size_t neededBlocks = gpu.getBlocksForMemory(getMaxNeededMemory());
    if (neededBlocks > memoryBlocks.size()) {
        reservedBlocks = neededBlocks - memoryBlocks.size();
        gpu.reserve(reservedBlocks, *this);
    }
    while (memoryBlocks.size() < neededBlocks) {
        gpu.allocateBlock(*this);
    }
}

void EmbeddingRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }

    if (tokens.empty()) {
        isFinished = true;
        return;
    }

    size_t slot = memoryBlocks[0];
    size_t embeddingSize = model.getEmbeddingSize();

    for (size_t i = 0; i < tokens.size(); i++) {
        vector<string> currentTokenVector = { tokens[i] };
        model.generateTokenEmbedding(currentTokenVector, gpu, slot, 0);

        for (size_t j = 0; j < embeddingSize; j++) {
            accumulatedSum[j] += gpu.readFromBlock(slot, j);
        }
    }

    for (size_t i = 0; i < embeddingSize; i++) {
        int avg = accumulatedSum[i] / static_cast<int>(tokens.size());
        gpu.writeToBlock(slot, embeddingSize + i, avg);
    }

    isFinished = true;
}

bool EmbeddingRequest::finished() const {
    return isFinished;
}

string EmbeddingRequest::getOutput() const {
    if (tokens.empty() || accumulatedSum.empty()) {
        return "[]";
    }

    int count = static_cast<int>(tokens.size());
    string result = "[" + to_string(accumulatedSum[0] / count);
    for (size_t i = 1; i < accumulatedSum.size(); i++) {
        result += ", " + to_string(accumulatedSum[i] / count);
    }
    result += "]";
    return result;
}

size_t EmbeddingRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}