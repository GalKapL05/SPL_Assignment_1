#pragma once

#include <iostream>

#include "GPU.h"
#include "Request.h"
#include "SchedulingPolicy.h"

using std::cout;
using std::endl;
using std::vector;
using std::string;

struct GPUConfig{
    size_t memoryBytes;
};

struct ModelConfig{
    string id;
    size_t embeddingSize;
};

enum class RequestType { Generative, TextEmbedder };

struct RequestConfig{
    size_t id;
    RequestType type;
    string modelId;
    int admissionCycle;
    string prompt;
    string scriptedOutput;
    size_t maxOutputTokens;
};

struct Config{
    size_t blockSize;
    vector<GPUConfig> gpus;
    vector<ModelConfig> models;
    vector<RequestConfig> requests;
    string policy;
};

class Model;
class SchedulingPolicy;

class Server{
private:
    Config readConfig(string configPath) const;

    vector<GPU> gpus;
    vector<Model*> models;
    vector<Request*> nonSubmittedRequests;
    vector<Request*> waitingRequests;
    vector<Request*> activeRequests;
    vector<size_t> activeGpuIndices;
    SchedulingPolicy* policy;
    size_t currentCycle;

    void clearAll();
    const Model* findModelById(const string& id) const;

public:
    Server(string configPath);

    ~Server();                                   
    Server(const Server& other);                
    Server& operator=(const Server& other);     
    Server(Server&& other);            
    Server& operator=(Server&& other); 

    bool step();
    void serve();
};