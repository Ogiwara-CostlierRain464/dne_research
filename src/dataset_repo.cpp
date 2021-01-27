#include "dataset_repo.h"

using namespace std;
using namespace boost;
using UGraph = DatasetLoader::UGraph;

void DatasetRepo::load(
        Dataset dataset,
        double train_ratio,
        DatasetLoader::UGraph &out_graph,
        std::unordered_map<size_t, size_t> &out_T,
        std::vector<size_t> &out_answer) {

    std::unordered_map<size_t, std::vector<size_t>> groups;
    std::unordered_map<size_t, std::vector<size_t>> nodes;
    switch (dataset) {
        case Dataset::Flickr: {
            auto N = 80513;
            auto C = 195;
            DatasetLoader::from_my_format(
                    "../dataset/flickr.txt",
                    N, C, out_graph, groups, nodes);
        }
        default: {
            assert(false);
        }
    }

    clean(train_ratio, out_graph, groups, nodes, out_T, out_answer);

}

void DatasetRepo::clean(
        double train_ratio, UGraph &graph,
        std::unordered_map<size_t, std::vector<size_t>> &groups,
        std::unordered_map<size_t, std::vector<size_t>> &nodes,
        std::unordered_map<size_t, size_t> &out_T,
        vector<size_t> &out_answer) {


    // まずデータをきれいにした後に、値を返す必要がある。
    // graph, nodes, groupsのそれぞれから該当するnodeについての情報を削除

    for(size_t v = 0; v < nodes.size(); ++v){
        // まずは、二つ以上のclassが割り当てられたnodeを削除
        if(nodes[v].size() >= 2){
            clear_vertex(v, graph);
            remove_vertex(v, graph);

            for(auto v_join_group: nodes[v]){
                assert(std::count(groups[v_join_group].begin(),groups[v_join_group].end(), v ) != 0);

                groups[v_join_group].erase(std::remove(groups[v_join_group].begin(), groups[v_join_group].end(), v), groups[v_join_group].end());
            }
            nodes.erase(v);
        }
        // 次に、一つもクラスが割り当てられていないnodeを削除
        if(nodes[v].empty()){
            clear_vertex(v, graph);
            remove_vertex(v, graph);
        }
    }

    // 最後に、(クラスが割り当てられていても)孤立したnodeを削除
    typedef boost::graph_traits<UGraph> GraphTraits;
    typename GraphTraits::vertex_iterator v, v_end;
    for(boost::tie(v, v_end) = boost::vertices(graph); v != v_end ; ++v){
        auto iter_pair = boost::out_edges(*v, graph);
        auto num_edges = std::distance(iter_pair.first, iter_pair.second);
        if(num_edges == 0){
            // もし本当に0なら、clean_vertexをする必要はない。
            boost::remove_vertex(*v, graph);

            for(auto v_join_group: nodes[*v]){
                groups[v_join_group].erase(
                        std::remove(groups[v_join_group].begin(),groups[v_join_group].end(), *v),
                        groups[v_join_group].end()
                        );
            }
            nodes.erase(*v);
        }
    }

    // 実は上のアルゴリズムでは孤立したnodeを完全には消せていない。しかしながら、多少のずさんさは許されるであろう。
}
