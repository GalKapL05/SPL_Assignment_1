#include <iostream>
#include "Server.h"

using namespace std;

int main(int argc, char** argv){
    if(argc < 2){
        cerr << "usage: " << argv[0] << " <config.json>" << endl;
        return 1;
    }
    Server s(argv[1]);
    s.serve();
    return 0;
}
