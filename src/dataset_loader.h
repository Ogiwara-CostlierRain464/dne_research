#ifndef DNE_DATASET_LOADER_H
#define DNE_DATASET_LOADER_H

#include <boost/graph/adjacency_list.hpp>
#include <string>

class DatasetLoader {
public:
    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> UGraph;


    /**
     * .matを変更して独自formatにしたファイルからよみこみ。
     * `########` で各nodeの属するclass, nodeごとのつながりを区別
     * @param file_name
     * @param node_num
     * @param group_num
     * @param out_graph
     * @param out_groups groupとそれに属するnodeのペア、の辞書
     * @param out_nodes nodeとそれが属するgroupのペア、の辞書
     */
    static void from_my_format(
            const std::string &file_name,
            size_t node_num,
            size_t group_num,
            UGraph &out_graph,
            std::unordered_map<size_t, std::vector<size_t>> &out_groups,
            std::unordered_map<size_t, std::vector<size_t>> &out_nodes
    );

    static void from_separate_format(
            const std::string &class_file_name,
            const std::string &edge_file_name,
            size_t node_num,
            size_t group_num,
            UGraph &out_graph,
            std::unordered_map<size_t, std::vector<size_t>> &out_groups,
            std::unordered_map<size_t, std::vector<size_t>> &out_nodes
    );

  static void from_DBLP_format(
    const std::string &class_file_name,
    const std::string &edge_file_name,
    size_t node_num,
    size_t group_num,
    UGraph &out_graph,
    std::unordered_map<size_t, std::vector<size_t>> &out_groups,
    std::unordered_map<size_t, std::vector<size_t>> &out_nodes
  );
};


#endif //DNE_DATASET_LOADER_H
