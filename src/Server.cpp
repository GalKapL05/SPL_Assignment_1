#include <fstream>
#include <stdexcept>
#include "json.hpp"

#include "Server.h"

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

Server::Server(string configPath) :     
    gpus(),
    models(),
    nonSubmittedRequests(),
    waitingRequests(),
    activeRequests(),
    policy(nullptr) {
    Config conf = readConfig(configPath);

    if (conf.policy == "shortestJob" || conf.policy == "ShortestJob") {
        policy = new ShortestJobPolicy();
    } else if (conf.policy == "longestPrefix" || conf.policy == "LongestPrefix") {
        policy = new LongestPrefixPolicy();
    } else {
        policy = new ArrivalOrderPolicy();
    }

    for (size_t i = 0; i < conf.gpus.size(); i++) {
        gpus.push_back(GPU(conf.gpus[i].memoryBytes, conf.blockSize));
    }

    for (size_t i = 0; i < conf.models.size(); i++) {
        models.push_back(new Model(conf.models[i].id, conf.models[i].embeddingSize));
    }

    for (size_t i = 0; i < conf.requests.size(); i++) {
        const RequestConfig& reqConf = conf.requests[i];

        const Model* targetModel = nullptr;
        for (size_t j = 0; j < models.size(); j++) {
            if (models[j]->getModelId() == reqConf.modelId) {
                targetModel = models[j];
                break;
            }
        }

        if (targetModel != nullptr) {
            if (reqConf.type == RequestType::Generative) {
                nonSubmittedRequests.push_back(new GenerativeRequest(
                    static_cast<int>(reqConf.id),
                    *targetModel,
                    reqConf.admissionCycle,
                    reqConf.prompt,
                    reqConf.scriptedOutput,
                    reqConf.maxOutputTokens
                ));
            } else {
                nonSubmittedRequests.push_back(new EmbeddingRequest(
                    static_cast<int>(reqConf.id),
                    *targetModel,
                    reqConf.admissionCycle,
                    reqConf.prompt
                ));
            }
        }
    }
}

void Server::clearAll() {
    for (size_t i = 0; i < models.size(); i++) {
        delete models[i];
    }
    models.clear();

    for (size_t i = 0; i < nonSubmittedRequests.size(); i++) {
        delete nonSubmittedRequests[i];
    }
    nonSubmittedRequests.clear();

    for (size_t i = 0; i < waitingRequests.size(); i++) {
        delete waitingRequests[i];
    }
    waitingRequests.clear();

    for (size_t i = 0; i < activeRequests.size(); i++) {
        delete activeRequests[i];
    }
    activeRequests.clear();

    delete policy;
    policy = nullptr;
}

Server::~Server() {
    clearAll();
}

Server::Server(const Server& other) : 
    gpus(other.gpus),
    models(),
    nonSubmittedRequests(),
    waitingRequests(),
    activeRequests(),
    policy(nullptr) {

    for (size_t i = 0; i < other.models.size(); ++i) {
        models.push_back(new Model(*other.models[i]));
    }

    if (other.policy != nullptr) {
        policy = other.policy->clone();
    }

    auto findMyModel = [this](const string& id) -> const Model* {
        for (size_t i = 0; i < models.size(); i++) {
            if (models[i]->getModelId() == id) {
                return models[i];
            }
        }
        return nullptr;
    };

    for (size_t i = 0; i < other.nonSubmittedRequests.size(); i++) {
        const Request* req = other.nonSubmittedRequests[i];
        const Model* myModel = findMyModel(req->getModel().getModelId());
        if (myModel != nullptr) {
            nonSubmittedRequests.push_back(req->clone(*myModel));
        }
    }

    for (size_t i = 0; i < other.waitingRequests.size(); i++) {
        const Request* req = other.waitingRequests[i];
        const Model* myModel = findMyModel(req->getModel().getModelId());
        if (myModel != nullptr) {
            waitingRequests.push_back(req->clone(*myModel));
        }
    }

    for (size_t i = 0; i < other.activeRequests.size(); i++) {
        const Request* req = other.activeRequests[i];
        const Model* myModel = findMyModel(req->getModel().getModelId());
        if (myModel != nullptr) {
            activeRequests.push_back(req->clone(*myModel));
        }
    }
}

Server& Server::operator=(const Server& other) {
    if (this != &other) {
        clearAll();

        gpus = other.gpus;

        for (size_t i = 0; i < other.models.size(); i++) {
            models.push_back(new Model(*other.models[i]));
        }

        if (other.policy != nullptr) {
            policy = other.policy->clone();
        }

        auto findMyModel = [this](const string& id) -> const Model* {
            for (size_t i = 0; i < models.size(); i++) {
                if (models[i]->getModelId() == id) {
                    return models[i];
                }
            }
            return nullptr;
        };

        for (size_t i = 0; i < other.nonSubmittedRequests.size(); i++) {
            const Request* req = other.nonSubmittedRequests[i];
            const Model* myModel = findMyModel(req->getModel().getModelId());
            if (myModel != nullptr) {
                nonSubmittedRequests.push_back(req->clone(*myModel));
            }
        }

        for (size_t i = 0; i < other.waitingRequests.size(); i++) {
            const Request* req = other.waitingRequests[i];
            const Model* myModel = findMyModel(req->getModel().getModelId());
            if (myModel != nullptr) {
                waitingRequests.push_back(req->clone(*myModel));
            }
        }

        for (size_t i = 0; i < other.activeRequests.size(); i++) {
            const Request* req = other.activeRequests[i];
            const Model* myModel = findMyModel(req->getModel().getModelId());
            if (myModel != nullptr) {
                activeRequests.push_back(req->clone(*myModel));
            }
        }
    }
    return *this;
}

Server::Server(Server&& other) noexcept
    : gpus(std::move(other.gpus)),
      models(std::move(other.models)),
      nonSubmittedRequests(std::move(other.nonSubmittedRequests)),
      waitingRequests(std::move(other.waitingRequests)),
      activeRequests(std::move(other.activeRequests)),
      policy(other.policy) {
    other.policy = nullptr;
}

Server& Server::operator=(Server&& other) noexcept {
    if (this != &other) {
        clearAll();

        gpus = std::move(other.gpus);
        models = std::move(other.models);
        nonSubmittedRequests = std::move(other.nonSubmittedRequests);
        waitingRequests = std::move(other.waitingRequests);
        activeRequests = std::move(other.activeRequests);
        
        policy = other.policy;
        other.policy = nullptr;
    }
    return *this;
}

bool Server::step() {
    static int currentCycle = 0;

    vector<Request*> remainingNonSubmitted;
    for (size_t i = 0; i < nonSubmittedRequests.size(); i++) {
        if (nonSubmittedRequests[i]->getAdmissionCycle() <= currentCycle) {
            waitingRequests.push_back(nonSubmittedRequests[i]);
        } else {
            remainingNonSubmitted.push_back(nonSubmittedRequests[i]);
        }
    }
    nonSubmittedRequests = remainingNonSubmitted;

    if (!waitingRequests.empty() && policy != nullptr) {
        vector<const Request*> eligible;
        eligible.reserve(waitingRequests.size());
        for (size_t i = 0; i < waitingRequests.size(); i++) {
            eligible.push_back(waitingRequests[i]);
        }

        vector<int> scheduleOrder = policy->choose(eligible, gpus);
        vector<bool> scheduled(waitingRequests.size(), false);
        for (size_t i = 0; i < scheduleOrder.size(); i++) {
            int reqIdx = scheduleOrder[i];
            Request* req = waitingRequests[reqIdx];

            for (size_t j = 0; j < gpus.size(); j++) {
                size_t neededBlocks = gpus[j].getBlocksForMemory(req->getMaxNeededMemory());
                if (gpus[j].availableSlots() >= neededBlocks) {
                    req->reserveMemory(gpus[j]);
                    activeRequests.push_back(req);
                    scheduled[reqIdx] = true;
                    break;
                }
            }
        }

        vector<Request*> newWaiting;
        for (size_t i = 0; i < waitingRequests.size(); i++) {
            if (!scheduled[i]) {
                newWaiting.push_back(waitingRequests[i]);
            }
        }
        waitingRequests = newWaiting;
    }

    vector<Request*> remainingActive;
    for (size_t i = 0; i < activeRequests.size(); i++) {
        Request* req = activeRequests[i];
        
        if (!gpus.empty()) {
            req->act(gpus[0]);
        }

        if (req->finished()) {
            if (!gpus.empty()) {
                req->deallocateMemory(gpus[0]);
            }
            delete req;
        } else {
            remainingActive.push_back(req);
        }
    }
    activeRequests = remainingActive;

    ++currentCycle;

    return !(nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty());
}

void Server::serve() {
    while (step()) {}

    for (size_t i = 0; i < models.size(); i++) {
        delete models[i];
    }
    models.clear();

    delete policy;
    policy = nullptr;
}