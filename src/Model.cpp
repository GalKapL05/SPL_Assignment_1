#include <cstdint>
#include "Model.h"

Model::Model(const string &id, size_t embeddingSize) :
    modelId(id), embeddingSize(embeddingSize) {}

vector<string> Model::tokenizer(string text) const {
    vector<string> tokens;
    string currentToken = "";

    for (char c : text) {
        if (c == ' ') {
            if (!currentToken.empty()) {
                tokens.push_back(currentToken);
                currentToken.clear();
            }
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
            currentToken += c;
        }
    }

    if (!currentToken.empty()) {
        tokens.push_back(currentToken);
    }

    return tokens;
}


uint64_t Model::mixByte(uint64_t state, unsigned char byte) const {
    state ^= byte;
    state *= 1099511628211ULL;
    return state;
}

uint64_t Model::mixString(uint64_t state, const string& text) const {
    const uint64_t length = static_cast<uint64_t>(text.size());
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        state = mixByte(state, static_cast<unsigned char>((length >> shift) & 0xFFULL));
    }
    for (unsigned char byte : text) {
        state = mixByte(state, byte);
    }
    return state;
}

// Produces reproducible synthetic vectors for the simulation.
// The constants come from FNV-1a and SplitMix64; these are not real text embeddings.
void Model::generateTokenEmbedding(const vector<string>& tokens, GPU& gpu, size_t slot, size_t offset) const {
    // The tokens are hashed joined by single spaces (a token never contains a space).
    string token;
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            token += " ";
        }
        token += tokens[i];
    }
    // Length-prefix both fields before hashing their bytes with FNV-1a.
    uint64_t state = 14695981039346656037ULL;

    state = mixString(state, modelId);
    state = mixString(state, token);



    // Generate one deterministic value per dimension.
    for (size_t i = 0; i < embeddingSize; ++i) {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t value = state;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        value ^= value >> 31;
        gpu.writeToBlock(slot, offset + i, static_cast<int>(value % 2001ULL) - 1000);
    }
}

const string &Model::getModelId() const
{
    return modelId;
}

const size_t &Model::getEmbeddingSize() const
{
    return embeddingSize;
}
