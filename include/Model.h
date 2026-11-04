#ifndef MODEL_H_
#define MODEL_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class Request;
class GPU;

class Model {
public:
    std::vector<std::string> tokenizer(std::string text) const;
    Model(const std::string& id, std::size_t embeddingSize);
    void generateTokenEmbedding(const std::vector<std::string>&, GPU&, size_t, size_t) const;
private:
    std::string modelId;
    std::size_t embeddingSize;

    std::uint64_t mixByte(std::uint64_t state, unsigned char byte) const;
    std::uint64_t mixString(std::uint64_t state, const std::string& text) const;
};


#endif