#ifndef SERVER_H_
#define SERVER_H_

#include <vector>
#include <string>
#include "Request.h"
#include "GPU.h"

struct GPUConfig{
    std::size_t memoryBytes;
};

struct ModelConfig{
    std::string id;
    std::size_t embeddingSize;
};

enum class RequestType { Generative, TextEmbedder };

struct RequestConfig{
    std::size_t id;
    RequestType type;
    std::string modelId;
    int admissionCycle;
    std::string prompt;
    std::string scriptedOutput;
    size_t maxOutputTokens;
};

struct Config{
    std::size_t blockSize;
    std::vector<GPUConfig> gpus;
    std::vector<ModelConfig> models;
    std::vector<RequestConfig> requests;
    std::string policy;
};

class Model;
class SchedulingPolicy;

class Server{
private:
    Config readConfig(std::string configPath) const;

    std::vector<GPU> gpus;
    std::vector<Model*> models;
    std::vector<Request*> nonSubmittedRequests;
    std::vector<Request*> waitingRequests;
    std::vector<Request*> activeRequests;
    SchedulingPolicy* policy;
public:
    Server(std::string configPath);
    bool step();
    void serve();
};

#endif