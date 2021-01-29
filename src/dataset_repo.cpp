#include "dataset_repo.h"
#include <fstream>
#include "param.h"
#include <unistd.h>

using namespace std;
using namespace boost;
using UGraph = DatasetLoader::UGraph;

void DatasetRepo::load(
        Dataset dataset,
        DatasetLoader::UGraph &out_graph,
        std::unordered_map<size_t, size_t> &out_T,
        std::vector<size_t> &out_answer) {

    std::unordered_map<size_t, std::vector<size_t>> groups;
    std::unordered_map<size_t, std::vector<size_t>> nodes;
    switch (dataset) {
        case Dataset::Karate: {
            auto N = 34;
            auto C = 2;
            DatasetLoader::from_my_format(
                    "../dataset/karate.txt",
                    N, C, out_graph, groups, nodes);
        }
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

    clean(out_graph, groups, nodes, out_T, out_answer);



}

void DatasetRepo::clean(
        UGraph &graph,
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
    out_T = std::unordered_map<size_t, size_t>();
    for(size_t group_id = 0; group_id < groups.size(); ++group_id){
        auto group_size = groups[group_id].size();
        assert(group_size >= 1);

        auto sample_count = ceil(group_size * FLAGS_train_ratio);
        assert(sample_count >= 1);

        for(size_t j = 0; j < sample_count; ++j){
            auto node_in_i = groups[group_id][j];
            out_T[node_in_i] = group_id;
        }
    }

    out_answer = std::vector<size_t>();
    out_answer.resize(num_vertices(graph));
    for(size_t group_id = 0; group_id < groups.size(); ++group_id){
        for(auto node_id : groups[group_id]){
            out_answer[node_id] = group_id;
        }
    }
}

template<class Matrix>
void DatasetRepo::saveMatrix(const string &filename, const Matrix &mat) {
    ofstream out(filename, ios::out | ios::binary | ios::trunc);
    assert(out.is_open());
    typename Matrix::Index rows=mat.rows(), cols=mat.cols();
    out.write(reinterpret_cast<char*>(&rows), sizeof(typename Matrix::Index));
    out.write(reinterpret_cast<char*>(&cols), sizeof(typename Matrix::Index));
    out.write(reinterpret_cast<const char*>(mat.data()), rows*cols*static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)) );
    out.close();
}

template<class Matrix>
void DatasetRepo::loadMatrix(const string &filename, Matrix &mat) {
    ifstream in(filename, ios::in | ios::binary);
    assert(in.is_open());
    typename Matrix::Index rows=0, cols=0;
    in.read(reinterpret_cast<char*>(&rows),sizeof(typename Matrix::Index));
    in.read(reinterpret_cast<char*>(&cols),sizeof(typename Matrix::Index));
    mat.resize(rows, cols);
    in.read(reinterpret_cast<char*>(mat.data()), rows*cols*static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)) );
    in.close();
}

void
DatasetRepo::loadS(DatasetRepo::Dataset dataset, Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &S,
                   std::unordered_map<size_t, size_t> &out_T, vector<size_t> &out_answer) {
    UGraph g;
    load(dataset, g, out_T, out_answer);

    const char *matrix_data_path;
    switch (dataset) {
        case Karate:
            matrix_data_path = "../dataset/karate.bin";
            break;
        case YouTube:
            break;
        case Flickr:
            break;
        case Wiki:
            break;
    }

    if(access( matrix_data_path, F_OK )){
        // load
    }else{
        // save
    }
}

template<class SparseMatrix>
void DatasetRepo::saveSparseMatrix(const string &filename, const SparseMatrix &mat) {
    assert(mat.isCompressed() == true);
    std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    assert(out.is_open());
    typename SparseMatrix::Index rows, cols, nnzs, outS, innS;
    rows = mat.rows();
    cols = mat.cols();
    nnzs = mat.nonZeros();
    outS = mat.outerSize();
    innS = mat.innerSize();

    out.write(reinterpret_cast<char*>(&rows), sizeof(typename SparseMatrix::Index));
    out.write(reinterpret_cast<char*>(&cols), sizeof(typename SparseMatrix::Index));
    out.write(reinterpret_cast<char*>(&nnzs), sizeof(typename SparseMatrix::Index));
    out.write(reinterpret_cast<char*>(&outS), sizeof(typename SparseMatrix::Index));
    out.write(reinterpret_cast<char*>(&innS), sizeof(typename SparseMatrix::Index));

    auto sizeIndexS = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::StorageIndex));
    auto sizeScalar = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Scalar      ));
    out.write(reinterpret_cast<const char*>(mat.valuePtr()),       sizeScalar * nnzs);
    out.write(reinterpret_cast<const char*>(mat.outerIndexPtr()),  sizeIndexS  * outS);
    out.write(reinterpret_cast<const char*>(mat.innerIndexPtr()),  sizeIndexS  * nnzs);

    out.close();
}

template<class SparseMatrix>
void DatasetRepo::loadSparseMatrix(const string &filename, SparseMatrix &mat) {
    std::ifstream in(filename, std::ios::binary | std::ios::in);
    assert(in.is_open());
    typename SparseMatrix::Index rows, cols, nnz, inSz, outSz;
    auto sizeScalar = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Scalar      ));
    auto sizeIndex  = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Index       ));
    auto sizeIndexS = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::StorageIndex));

    in.read(reinterpret_cast<char*>(&rows ), sizeIndex);
    in.read(reinterpret_cast<char*>(&cols ), sizeIndex);
    in.read(reinterpret_cast<char*>(&nnz  ), sizeIndex);
    in.read(reinterpret_cast<char*>(&outSz), sizeIndex);
    in.read(reinterpret_cast<char*>(&inSz ), sizeIndex);

    mat.resize(rows, cols);
    mat.makeCompressed();
    mat.resizeNonZeros(nnz);

    in.read(reinterpret_cast<char*>(mat.valuePtr())     , sizeScalar * nnz  );
    in.read(reinterpret_cast<char*>(mat.outerIndexPtr()), sizeIndexS * outSz);
    in.read(reinterpret_cast<char*>(mat.innerIndexPtr()), sizeIndexS * nnz );

    mat.finalize();
    in.close();
}
