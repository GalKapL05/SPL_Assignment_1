#include <cstdint>
#include "Model.h"

std::uint64_t Model::mixByte(std::uint64_t state, unsigned char byte) const {
    state ^= byte;
    state *= 1099511628211ULL;
    return state;
}

std::uint64_t Model::mixString(std::uint64_t state, const std::string& text) const {
    const std::uint64_t length = static_cast<std::uint64_t>(text.size());
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
void Model::generateTokenEmbedding(const std::vector<std::string>& tokens, GPU& gpu, size_t slot,
size_t offset) const {
    // The tokens are hashed joined by single spaces (a token never contains a space).
    std::string token;
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        if (i > 0) {
            token += " ";
        }
        token += tokens[i];
    }
    // Length-prefix both fields before hashing their bytes with FNV-1a.
    std::uint64_t state = 14695981039346656037ULL;

    state = mixString(state, modelId);
    state = mixString(state, token);



    // Generate one deterministic value per dimension.
    for (std::size_t i = 0; i < embeddingSize; ++i) {
        state += 0x9E3779B97F4A7C15ULL;
        std::uint64_t value = state;
        value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
        value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
        value ^= value >> 31;
        gpu.writeToBlock(slot, offset + i, static_cast<int>(value % 2001ULL) - 1000);
    }
}