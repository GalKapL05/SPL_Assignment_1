#include <fstream>
#include <stdexcept>
#include "json.hpp"

#include "Server.h"

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    ifstream input(configPath);
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
    activeGpuIndices(),
    policy(nullptr),
    currentCycle(0) {

    Config config = readConfig(configPath);

    if (config.policy == "shortestJob") {
        policy = new ShortestJobPolicy();
    } else if (config.policy == "longestPrefix") {
        policy = new LongestPrefixPolicy();
    } else {
        policy = new ArrivalOrderPolicy();
    }

    for (size_t i = 0; i < config.gpus.size(); i++) {
        size_t blocksNum = config.gpus[i].memoryBytes / config.blockSize;
        gpus.push_back(GPU(blocksNum, config.blockSize));
    }

    for (size_t i = 0; i < config.models.size(); i++) {
        models.push_back(new Model(config.models[i].id, config.models[i].embeddingSize));
    }

    for (size_t i = 0; i < config.requests.size(); i++) {
        const RequestConfig& requestConfig = config.requests[i];

        const Model* targetModel = nullptr;
        for (size_t j = 0; j < models.size(); j++) {
            if (models[j]->getModelId() == requestConfig.modelId) {
                targetModel = models[j];
                break;
            }
        }

        if (targetModel != nullptr) {
            if (requestConfig.type == RequestType::Generative) {
                nonSubmittedRequests.push_back(new GenerativeRequest(
                    static_cast<int>(requestConfig.id),
                    *targetModel,
                    requestConfig.admissionCycle,
                    requestConfig.prompt,
                    requestConfig.scriptedOutput,
                    requestConfig.maxOutputTokens
                ));
            } else {
                nonSubmittedRequests.push_back(new EmbeddingRequest(
                    static_cast<int>(requestConfig.id),
                    *targetModel,
                    requestConfig.admissionCycle,
                    requestConfig.prompt
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
    activeGpuIndices.clear();

    delete policy;
    policy = nullptr;
}

const Model* Server::findModelById(const string& id) const {
    for (size_t i = 0; i < models.size(); i++) {
        if (models[i]->getModelId() == id) {
            return models[i];
        }
    }
    return nullptr;
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
    activeGpuIndices(other.activeGpuIndices),
    policy(nullptr),
    currentCycle(other.currentCycle) {

    for (size_t i = 0; i < other.models.size(); i++) {
        models.push_back(new Model(*other.models[i]));
    }

    if (other.policy != nullptr) {
        policy = other.policy->clone();
    }

    for (size_t i = 0; i < other.nonSubmittedRequests.size(); i++) {
        const Request* req = other.nonSubmittedRequests[i];
        const Model* myModel = findModelById(req->getModel().getModelId());
        if (myModel != nullptr) {
            nonSubmittedRequests.push_back(req->clone(*myModel));
        }
    }

    for (size_t i = 0; i < other.waitingRequests.size(); i++) {
        const Request* req = other.waitingRequests[i];
        const Model* myModel = findModelById(req->getModel().getModelId());
        if (myModel != nullptr) {
            waitingRequests.push_back(req->clone(*myModel));
        }
    }

    for (size_t i = 0; i < other.activeRequests.size(); i++) {
        const Request* req = other.activeRequests[i];
        const Model* myModel = findModelById(req->getModel().getModelId());
        if (myModel != nullptr) {
            activeRequests.push_back(req->clone(*myModel));
        }
    }
}

Server& Server::operator=(const Server& other) {
    if (this != &other) {
        clearAll();

        gpus = other.gpus;
        currentCycle = other.currentCycle;
        activeGpuIndices = other.activeGpuIndices;

        for (size_t i = 0; i < other.models.size(); i++) {
            models.push_back(new Model(*other.models[i]));
        }

        if (other.policy != nullptr) {
            policy = other.policy->clone();
        }

        for (size_t i = 0; i < other.nonSubmittedRequests.size(); i++) {
            const Request* req = other.nonSubmittedRequests[i];
            const Model* myModel = findModelById(req->getModel().getModelId());
            if (myModel != nullptr) {
                nonSubmittedRequests.push_back(req->clone(*myModel));
            }
        }

        for (size_t i = 0; i < other.waitingRequests.size(); i++) {
            const Request* req = other.waitingRequests[i];
            const Model* myModel = findModelById(req->getModel().getModelId());
            if (myModel != nullptr) {
                waitingRequests.push_back(req->clone(*myModel));
            }
        }

        for (size_t i = 0; i < other.activeRequests.size(); i++) {
            const Request* req = other.activeRequests[i];
            const Model* myModel = findModelById(req->getModel().getModelId());
            if (myModel != nullptr) {
                activeRequests.push_back(req->clone(*myModel));
            }
        }
    }
    return *this;
}

Server::Server(Server&& other)
    : gpus(move(other.gpus)),
      models(move(other.models)),
      nonSubmittedRequests(move(other.nonSubmittedRequests)),
      waitingRequests(move(other.waitingRequests)),
      activeRequests(move(other.activeRequests)),
      activeGpuIndices(move(other.activeGpuIndices)),
      policy(other.policy),
      currentCycle(other.currentCycle) {
    other.policy = nullptr;
    other.currentCycle = 0;
}

Server& Server::operator=(Server&& other) {
    if (this != &other) {
        clearAll();

        gpus = move(other.gpus);
        models = move(other.models);
        nonSubmittedRequests = move(other.nonSubmittedRequests);
        waitingRequests = move(other.waitingRequests);
        activeRequests = move(other.activeRequests);
        activeGpuIndices = move(other.activeGpuIndices);
        
        policy = other.policy;
        currentCycle = other.currentCycle;

        other.policy = nullptr;
        other.currentCycle = 0;
    }
    return *this;
}

bool Server::step() {
    vector<Request*> remainingNonSubmitted;
    for (size_t i = 0; i < nonSubmittedRequests.size(); i++) {
        Request* req = nonSubmittedRequests[i];
        if (static_cast<size_t>(req->getAdmissionCycle()) <= currentCycle) {
            bool canEverFit = false;

            for (size_t j = 0; j < gpus.size(); j++) {
                size_t neededBlocks = gpus[j].getBlocksForMemory(req->getMaxNeededMemory());
                if (neededBlocks <= gpus[j].getBlocksNum()) {
                    canEverFit = true;
                    break;
                }
            }

            if (!canEverFit) {
                cout << "rejected " << req->getId() << " cycle=" << currentCycle << endl;
                delete req;
            } else {
                waitingRequests.push_back(req);
            }
        } else {
            remainingNonSubmitted.push_back(req);
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
            int requsetIndex = scheduleOrder[i];
            Request* request = waitingRequests[requsetIndex];

            int bestGpu = -1;
            size_t maxCachedBlocks = 0;
            size_t minNeededBlocks = static_cast<size_t>(-1);

            for (size_t j = 0; j < gpus.size(); j++) {
                PrefixMatch match = gpus[j].findLongestCachedPrefix(
                    request->getModel().getModelId(),
                    request->getModel().getEmbeddingSize(),
                    request->getTokens()
                );
                size_t cachedBlocks = match.blockIds.size();
                size_t neededBlocks = gpus[j].getBlocksForMemory(request->getNeededMemory(gpus[j]));

                if (gpus[j].availableSlots() >= neededBlocks) {
                    if (bestGpu == -1 || cachedBlocks > maxCachedBlocks) {
                        bestGpu = static_cast<int>(j);
                        maxCachedBlocks = cachedBlocks;
                        minNeededBlocks = neededBlocks;
                    } else if (cachedBlocks == maxCachedBlocks && neededBlocks < minNeededBlocks) {
                        bestGpu = static_cast<int>(j);
                        minNeededBlocks = neededBlocks;
                    }
                }
            }

            if (bestGpu != -1) {
                int longestPrefixGpu = -1;
                size_t longestMatch = 0;
                vector<size_t> blocksToCopy;

                for (size_t j = 0; j < gpus.size(); j++) {
                    if (static_cast<int>(j) != bestGpu && !request->getCacheableTokens().empty()) {
                        PrefixMatch match = gpus[j].findLongestCachedPrefix(
                            request->getModel().getModelId(),
                            request->getModel().getEmbeddingSize(),
                            request->getTokens()
                        );
                        if (match.matchedTokens > longestMatch) {
                            longestMatch = match.matchedTokens;
                            longestPrefixGpu = static_cast<int>(j);
                            blocksToCopy = match.blockIds;
                        }
                    }
                }

                if (longestPrefixGpu != -1 && longestMatch > maxCachedBlocks * (gpus[bestGpu].getBlockSize() / request->getModel().getEmbeddingSize())) {
                    gpus[bestGpu].copyMemoryFromOtherGPU(gpus[longestPrefixGpu], blocksToCopy);
                }

                request->reserveMemory(gpus[bestGpu]);
                activeRequests.push_back(request);
                activeGpuIndices.push_back(static_cast<size_t>(bestGpu));
                scheduled[requsetIndex] = true;
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
    vector<size_t> remainingGpuIndices;

    for (size_t i = 0; i < activeRequests.size(); i++) {
        Request* req = activeRequests[i];
        size_t gpuIdx = activeGpuIndices[i];

        req->act(gpus[gpuIdx]);

        if (req->finished()) {
            cout << "finished " << req->getId() 
                 << " cycle=" << currentCycle 
                 << " gpu=" << gpuIdx 
                 << ": " << req->getOutput() << "\n";

            req->deallocateMemory(gpus[gpuIdx]);
            delete req;
        } else {
            remainingActive.push_back(req);
            remainingGpuIndices.push_back(gpuIdx);
        }
    }
    activeRequests = remainingActive;
    activeGpuIndices = remainingGpuIndices;

    ++currentCycle;

    return !(nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty());
}

void Server::serve() {
    while (step()) {}

    size_t totalHits = 0;
    size_t totalEvictions = 0;
    for (size_t i = 0; i < gpus.size(); i++) {
        totalHits += gpus[i].getCacheHits();
        totalEvictions += gpus[i].getEvictions();
    }

    cout << "summary: cycles=" << currentCycle 
         << " cache_hits=" << totalHits 
         << " evictions=" << totalEvictions << endl;
}