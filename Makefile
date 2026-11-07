all: bin/server

bin/server: bin/Main.o bin/Server.o bin/Model.o bin/GPU.o bin/MemoryBlock.o bin/KVBlock.o bin/EmbeddingWorkspace.o bin/Request.o bin/GenerativeRequest.o bin/EmbeddingRequest.o bin/SchedulingPolicy.o bin/ArrivalOrderPolicy.o bin/ShortestJobPolicy.o bin/LongestPrefixPolicy.o
	g++ -o bin/server bin/Main.o bin/Server.o bin/Model.o bin/GPU.o bin/MemoryBlock.o bin/KVBlock.o bin/EmbeddingWorkspace.o bin/Request.o bin/GenerativeRequest.o bin/EmbeddingRequest.o bin/SchedulingPolicy.o bin/ArrivalOrderPolicy.o bin/ShortestJobPolicy.o bin/LongestPrefixPolicy.o

bin/Main.o: src/Main.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Main.cpp -o bin/Main.o

bin/Server.o: src/Server.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Server.cpp -o bin/Server.o

bin/Model.o: src/Model.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Model.cpp -o bin/Model.o

bin/GPU.o: src/GPU.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/GPU.cpp -o bin/GPU.o

bin/MemoryBlock.o: src/MemoryBlock.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/MemoryBlock.cpp -o bin/MemoryBlock.o

bin/KVBlock.o: src/KVBlock.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/KVBlock.cpp -o bin/KVBlock.o

bin/EmbeddingWorkspace.o: src/EmbeddingWorkspace.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/EmbeddingWorkspace.cpp -o bin/EmbeddingWorkspace.o

bin/Request.o: src/Request.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/Request.cpp -o bin/Request.o

bin/GenerativeRequest.o: src/GenerativeRequest.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/GenerativeRequest.cpp -o bin/GenerativeRequest.o

bin/EmbeddingRequest.o: src/EmbeddingRequest.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/EmbeddingRequest.cpp -o bin/EmbeddingRequest.o

bin/SchedulingPolicy.o: src/SchedulingPolicy.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/SchedulingPolicy.cpp -o bin/SchedulingPolicy.o

bin/ArrivalOrderPolicy.o: src/ArrivalOrderPolicy.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/ArrivalOrderPolicy.cpp -o bin/ArrivalOrderPolicy.o

bin/ShortestJobPolicy.o: src/ShortestJobPolicy.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/ShortestJobPolicy.cpp -o bin/ShortestJobPolicy.o

bin/LongestPrefixPolicy.o: src/LongestPrefixPolicy.cpp
	g++ -Wall -Weffc++ -std=c++11 -c -Iinclude -g src/LongestPrefixPolicy.cpp -o bin/LongestPrefixPolicy.o

clean:
	rm -f bin/*.o bin/server