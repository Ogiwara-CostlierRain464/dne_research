#include "dataset_loader.h"
#include <iostream>
#include <fstream>

using namespace std;

void DatasetLoader::from_my_format(
        const std::string &file_name,
        size_t node_num,
        size_t group_num,
        DatasetLoader::UGraph &out_graph,
        std::unordered_map<size_t, std::vector<size_t>> &out_groups,
        std::unordered_map<size_t, std::vector<size_t>> &out_nodes) {
    std::ifstream infile(file_name);
    assert(infile);
    typedef pair<int, int> Edge;
    vector<Edge> edges{};
    out_groups = std::unordered_map<size_t, std::vector<size_t>>{};
    out_nodes = std::unordered_map<size_t, std::vector<size_t>>{};

    string tmp_line{};
    bool group_flag = true;

    while(getline(infile, tmp_line)){
        if(tmp_line == "########"){
            // ここからはEdgeの記述
            group_flag = false;
            continue;
        }

        istringstream iss(tmp_line);
        // group modeにおいてはv1はnode id, v2はgroup id
        // Edge modeにおいてはv1, v2はそれぞれ隣接するvertex同志
        // ここで、my formatにおいてはidは1からはじまることに注意。
        int v1, v2, _;
        if(iss >> v1 >> v2 >> _){
            if(group_flag){
                assert(v2 <= group_num);
                out_groups[v2-1].push_back(v1-1);
                out_nodes[v1-1].push_back(v2-1);
            }else{
                edges.emplace_back(v1-1, v2-1);
            }

        }
    }

    out_graph = UGraph(edges.begin(), edges.end(), node_num);

}
