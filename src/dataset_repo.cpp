#include "dataset_repo.h"
#include <fstream>
#include "param.h"
#include <unistd.h>
#include <glog/logging.h>
#include <iostream>
#include <boost/serialization/unordered_map.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

using namespace std;
using namespace boost;
using UGraph = DatasetLoader::UGraph;

void DatasetRepo::load(
        Dataset dataset,
        DatasetLoader::UGraph &out_graph,
        std::unordered_map<size_t, size_t> &out_T,
        std::unordered_map<size_t, size_t> &out_answer) {
    assert(false);

    std::unordered_map<size_t, std::vector<size_t>> groups;
    std::unordered_map<size_t, std::vector<size_t>> nodes;


    string data_path;
    switch (dataset) {
        case Karate:
            data_path = "../dataset/karate/";
            break;
        case YouTube:
            data_path = "../dataset/youtube/";
            break;
        case Flickr:
            data_path = "../dataset/flickr/";
            break;
        case Wiki:
            data_path = "../dataset/wiki/";
            break;
        case BlogCatalog:
            data_path = "../dataset/blogcatalog/";
    }

    if(access((data_path + "S.matrix").c_str(), F_OK) != -1){
        // 保存済みのデータを返す
        assert(access((data_path + "groups.map").c_str(), F_OK) != -1);
        assert(access((data_path + "nodes.map").c_str(), F_OK) != -1);

    }



    // この段階で、保存されたデータはもうとれないか？
    // fileからgraphとclassを取得、それぞれcleanにする、Sを作る
    // この時点で完成しているはず

    // /karate 下に S.matrix  groups.map  nodes.map

    switch (dataset) {
        case Dataset::Karate: {
            auto N = 34;
            auto C = 2;
            DatasetLoader::from_my_format(
                    "../dataset/karate.txt",
                    N, C, out_graph, groups, nodes);
        }
        break;
        case Dataset::Flickr: {
            auto N = 80513;
            auto C = 195;
            DatasetLoader::from_my_format(
                    "../dataset/flickr.txt",
                    N, C, out_graph, groups, nodes);
        }
        break;
        case Dataset::YouTube: {
            auto N = 1138499;
            auto C = 47;
            DatasetLoader::from_my_format("../dataset/youtube.txt",
                    N, C, out_graph, groups, nodes);
        }
        break;
        case Dataset::BlogCatalog: {
            // Edge Number: 667932
            // x64.77 density. (or 32?)
            auto N = 10312;
            auto C = 39;
            DatasetLoader::from_my_format("../dataset/blogcatalog.txt",
                                          N, C, out_graph, groups, nodes);
        }
        break;
        case Dataset::Wiki: {
            // Edge Number: 17981
            // C = 16
            auto N = 2405;
            auto C = 17;
            DatasetLoader::from_separate_format("../dataset/Wiki_category.txt",
                                                "../dataset/Wiki_edgelist.txt",
                                                N, C, out_graph, groups, nodes);
        }
        break;
        default: {
            assert(false);
        }
    }

    clean(out_graph, groups, nodes, out_T, out_answer);



}

void DatasetRepo::load(DatasetRepo::Dataset dataset,
                       Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_S,
                       std::unordered_map<size_t, size_t> &out_T,
                       std::unordered_map<size_t, size_t> &out_answer,
                       size_t &out_class_num) {
    std::unordered_map<size_t, std::vector<size_t>> groups;
    std::unordered_map<size_t, std::vector<size_t>> nodes;
    loadAll(dataset, out_S, groups, nodes);

    out_T = std::unordered_map<size_t, size_t>();

    for(auto & group : groups){
        auto group_size = group.second.size();
        assert(group_size >= 1);

        auto sample_count = ceil(group_size * FLAGS_train_ratio);
        assert(sample_count >= 1);

        for(size_t j = 0; j < sample_count; ++j){
            auto node_in_i = group.second[j];
            out_T[node_in_i] = group.first;
        }
    }

    out_answer = std::unordered_map<size_t, size_t>();
    out_answer.reserve(nodes.size());
    for(auto & group : groups){
        for(auto node_id : group.second){
            out_answer[node_id] = group.first;
        }
    }

    out_class_num = groups.size();
}



void DatasetRepo::loadAll(DatasetRepo::Dataset dataset,
                          Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_S,
                          std::unordered_map<size_t, std::vector<size_t>> &out_groups,
                          std::unordered_map<size_t, std::vector<size_t>> &out_nodes) {

    string data_path;
    switch (dataset) {
        case Karate:
            data_path = "../dataset/karate/";
            break;
        case YouTube:
            data_path = "../dataset/youtube/";
            break;
        case Flickr:
            data_path = "../dataset/flickr/";
            break;
        case Wiki:
            data_path = "../dataset/wiki/";
            break;
        case BlogCatalog:
            data_path = "../dataset/blogcatalog/";
    }

    if(access((data_path + "S.matrix").c_str(), F_OK) != -1){
        // 保存済みのデータを返す
        assert(access((data_path + "groups.map").c_str(), F_OK) != -1);
        assert(access((data_path + "nodes.map").c_str(), F_OK) != -1);

        loadSparseMatrix(data_path + "S.matrix", out_S);
        std::ifstream groups_ifs(data_path + "groups.map");
        assert(groups_ifs.is_open());
        boost::archive::text_iarchive groups_ia(groups_ifs);
        groups_ia >> out_groups;

        std::ifstream nodes_ifs(data_path + "nodes.map");
        assert(nodes_ifs.is_open());
        boost::archive::text_iarchive nodes_ia(nodes_ifs);
        nodes_ia >> out_nodes;
    }else{
        DatasetLoader::UGraph graph;
        // データを取得して、cleanして、保存
        switch (dataset) {
            case Dataset::Karate: {
                auto N = 34;
                auto C = 2;
                DatasetLoader::from_my_format(
                        "../dataset/karate.txt",
                        N, C, graph, out_groups, out_nodes);
            }
                break;
            case Dataset::Flickr: {
                auto N = 80513;
                auto C = 195;
                DatasetLoader::from_my_format(
                        "../dataset/flickr.txt",
                        N, C, graph, out_groups, out_nodes);
            }
                break;
            case Dataset::YouTube: {
                auto N = 1138499;
                auto C = 47;
                DatasetLoader::from_my_format("../dataset/youtube.txt",
                                              N, C, graph, out_groups, out_nodes);
            }
                break;
            case Dataset::BlogCatalog: {
                // Edge Number: 667932
                // x64.77 density. (or 32?)
                auto N = 10312;
                auto C = 39;
                DatasetLoader::from_my_format("../dataset/blogcatalog.txt",
                                              N, C, graph, out_groups, out_nodes);
            }
                break;
            case Dataset::Wiki: {
                // Edge Number: 17981
                // C = 16
                auto N = 2405;
                auto C = 17;
                DatasetLoader::from_separate_format("../dataset/Wiki_category.txt",
                                                    "../dataset/Wiki_edgelist.txt",
                                                    N, C, graph, out_groups, out_nodes);
            }
                break;
            default: {
                assert(false);
            }
        }

        Eigen::SparseMatrix<double, 0, std::ptrdiff_t> A;
        clean(graph, out_groups, out_nodes, A);

        // Sを計算して保存
        out_S = (A + A * A) / 2;
        saveSparseMatrix(data_path + "S.matrix", out_S);

        std::ofstream groups_ofs(data_path + "groups.map");
        boost::archive::text_oarchive groups_oa(groups_ofs);
        groups_oa << out_groups;

        std::ofstream nodes_ofs(data_path + "nodes.map");
        boost::archive::text_oarchive nodes_oa(nodes_ofs);
        nodes_oa << out_nodes;
    }
}


void DatasetRepo::clean(
        UGraph &graph,
        std::unordered_map<size_t, std::vector<size_t>> &groups,
        std::unordered_map<size_t, std::vector<size_t>> &nodes,
        std::unordered_map<size_t, size_t> &out_T,
        std::unordered_map<size_t, size_t> &out_answer) {
    assert(false);



    // まずデータをきれいにした後に、値を返す必要がある。
    // graph, nodes, groupsのそれぞれから該当するnodeについての情報を削除

    std::cout << "Before erase" << std::endl;

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

    std::cout << "After erase" << std::endl;

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

    std::cout << "After erase 2" << std::endl;

    // ここまではtrain_ratioに依存しない

    // 実は上のアルゴリズムでは孤立したnodeを完全には消せていない。しかしながら、多少のずさんさは許されるであろう。
    out_T = std::unordered_map<size_t, size_t>();
    for(size_t group_id = 0; group_id < groups.size(); ++group_id){
        auto group_size = groups[group_id].size();
        // 前の処理で、このサイズが0になる可能性がある。
//        assert(group_size >= 1);

        auto sample_count = ceil(group_size * FLAGS_train_ratio);
//        assert(sample_count >= 1);

        for(size_t j = 0; j < sample_count; ++j){
            auto node_in_i = groups[group_id][j];
            out_T[node_in_i] = group_id;
        }
    }

    std::cout << "After erase 3" << std::endl;

    out_answer = std::unordered_map<size_t, size_t>();
    out_answer.reserve(num_vertices(graph));
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
DatasetRepo::loadS(DatasetRepo::Dataset dataset, Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_S,
                   std::unordered_map<size_t, size_t> &out_T, std::unordered_map<size_t, size_t> &out_answer) {
    assert(false);

    UGraph g;
    load(dataset, g, out_T, out_answer);

    string matrix_data_path;
    switch (dataset) {
        case Karate:
            matrix_data_path = "../dataset/karate.bin";
            break;
        case YouTube:
            matrix_data_path = "../dataset/youtube.bin";
            break;
        case Flickr:
            matrix_data_path = "../dataset/flickr.bin";
            break;
        case Wiki:
            matrix_data_path = "../dataset/wiki.bin";
            break;
        case BlogCatalog:
            matrix_data_path = "../dataset/blogcatalog.bin";
    }

    if(access( matrix_data_path.c_str(), F_OK ) != -1){
        // load
        LOG(INFO) << "loading: " << matrix_data_path;
        loadSparseMatrix(matrix_data_path, out_S);
    }else{
        // save
        LOG(INFO) << "save matrix S to: " << matrix_data_path;
        auto node_num = out_answer.size();

        Eigen::SparseMatrix<double, 0, std::ptrdiff_t> A(node_num, node_num);
        typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
        IndexMap index = get(boost::vertex_index, g);
        typedef boost::graph_traits<UGraph> GraphTraits;
        typename GraphTraits::edge_iterator ei, ei_end;
        for(tie(ei, ei_end) = edges(g); ei != ei_end; ++ei){
            auto sur = index[boost::source(*ei, g)];
            auto tar = index[boost::target(*ei, g)];

            A.coeffRef(sur, tar) = 1.0;
            A.coeffRef(tar, sur) = 1.0;
        }
        assert(A.isApprox(A.transpose()));

        #pragma omp parallel for
        for(size_t i = 0; i < node_num; ++i){
            A.row(i) /= A.row(i).sum();
        }

        out_S = (A + A * A) / 2;
        saveSparseMatrix(matrix_data_path, out_S);
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

void DatasetRepo::clean(UGraph &graph,
                        std::unordered_map<size_t, std::vector<size_t>> &groups,
                        std::unordered_map<size_t, std::vector<size_t>> &nodes,
                        Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_A) {

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

    // groups, nodesそれぞれに対し、空になったentryは削除
    for(auto it = groups.begin(); it != groups.end();){
        if(it->second.empty()){
            it = groups.erase(it);
        }else{
            ++it;
        }
    }

    for(auto it = nodes.begin(); it != nodes.end();){
        if(it->second.empty()){
            it = nodes.erase(it);
        }else{
            ++it;
        }
    }

  for(auto it = groups.begin(); it != groups.end(); ++it){
    assert(!it->second.empty());
  }

  for(auto it = nodes.begin(); it != nodes.end(); ++it) {
    assert(it->second.size() == 1);
  }

  // 全てのclass、全てのnodeに対するindexの再割り当てを行おう
  // node 1,3,5にはそれぞれ1,2,3を
  // class 1,2,4にはそれぞれ1,2,3を
  // これは必要か？問題の単純化には欠かせなさそう、もっとも何の意味があるかわからんが
  // これにより、node id は連続になり、group id もどうようにgroups.size() == 最後のclass idと同じに
  // nodes: 0: {}

  // さて、やはりA行列を作る前に、graphのnode_idの書き換えは必要

  std::unordered_map<size_t, size_t> node_id_mapping;
  std::unordered_map<size_t, size_t> group_id_mapping;

  std::unordered_map<size_t, std::vector<size_t>> true_nodes;
  std::unordered_map<size_t, std::vector<size_t>> true_groups;


  std::map<size_t, std::vector<size_t>> ordered_nodes(nodes.begin(), nodes.end());
  std::map<size_t, std::vector<size_t>> ordered_groups(groups.begin(), groups.end());

  // mappingを取得して、graphのidを書き換えなきゃ！

  size_t node_id_count = 0;
  for(const auto& it: ordered_nodes){
    node_id_mapping.emplace(it.first, node_id_count);
    true_nodes.emplace(node_id_count, it.second);
    ++node_id_count;
  }

  size_t group_id_count = 0;
  for(const auto& it: ordered_groups){
    group_id_mapping.emplace(it.first, group_id_count);
    true_groups.emplace(group_id_count, it.second);
    ++group_id_count;
  }

  // mappingの取得後、それぞれ書き直す
  for(auto& it: true_nodes){
    std::vector<size_t> clean_vector;
    clean_vector.resize(it.second.size());

    std::transform(
      it.second.begin(),
      it.second.end(),
      clean_vector.begin(), [&group_id_mapping](size_t group_id){
        return group_id_mapping[group_id];
      });

    it.second = clean_vector;
  }

  for(auto& it: true_groups){
    std::vector<size_t> clean_vector;
    clean_vector.resize(it.second.size());

    std::transform(
      it.second.begin(),
      it.second.end(),
      clean_vector.begin(), [&node_id_mapping](size_t node_id){
        return node_id_mapping[node_id];
      });

    it.second = clean_vector;
  }

  groups = true_groups;
  nodes = true_nodes;

  // ここで、単にAを作るときにだけmappingを施せばよいと気が付いた。
  // なのでここでAを作ろうか
  auto node_num = true_nodes.size();
  out_A = Eigen::SparseMatrix<double, 0, std::ptrdiff_t>(node_num, node_num);
  typedef boost::property_map<UGraph, boost::vertex_index_t>::type IndexMap;
  IndexMap index = get(boost::vertex_index, graph);
  typedef boost::graph_traits<UGraph> GraphTraits;
  typename GraphTraits::edge_iterator ei, ei_end;
  for(tie(ei, ei_end) = edges(graph); ei != ei_end; ++ei){
    auto sur = node_id_mapping[index[boost::source(*ei, graph)]];
    auto tar = node_id_mapping[index[boost::target(*ei, graph)]];

    out_A.coeffRef(sur, tar) = 1.0;
    out_A.coeffRef(tar, sur) = 1.0;
  }
  assert(out_A.isApprox(out_A.transpose()));

  #pragma omp parallel for
  for(size_t i = 0; i < node_num; ++i){
    out_A.row(i) /= out_A.row(i).sum();
  }
}
