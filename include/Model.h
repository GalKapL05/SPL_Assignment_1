#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>

#include "GPU.h"

using std::vector;
using std::string;

class GPU;

class Model {
public:
    Model(const string& id, size_t embeddingSize);

    vector<string> tokenizer(string text) const;
    void generateTokenEmbedding(const vector<string>&, GPU&, size_t, size_t) const;

    const string& getModelId() const;
    const size_t& getEmbeddingSize() const;

private:
    string modelId;
    size_t embeddingSize;

    uint64_t mixByte(uint64_t state, unsigned char byte) const;
    uint64_t mixString(uint64_t state, const string& text) const;
};