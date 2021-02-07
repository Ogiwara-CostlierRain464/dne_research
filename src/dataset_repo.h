#ifndef DNE_DATASET_REPO_H
#define DNE_DATASET_REPO_H

#include <boost/graph/adjacency_list.hpp>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "dataset_loader.h"

class DatasetRepo {
public:

    enum Dataset{
        YouTube,
        Flickr,
        Karate,
        Wiki,
        BlogCatalog
    };

    /**
     * [dataset]のデータを読み込み、out_パラメータに結果を格納する。
     * @param dataset
     * @param train_ratio
     * @param out_graph
     * @param out_T
     * @param out_answer
     */
    static void load(
            Dataset dataset,
            DatasetLoader::UGraph &out_graph,
            std::unordered_map<size_t, size_t> &out_T,
            std::unordered_map<size_t, size_t> &out_answer);

    static void load(
            Dataset dataset,
            Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_S,
            std::unordered_map<size_t, size_t> &out_T,
            std::unordered_map<size_t, size_t> &out_answer,
            size_t &out_class_num
            );

    /**
     * train_ratioによらず、すべてのデータを読み込む
     * cache済みの場合には、それをよみこむ
     * @param dataset
     * @param S
     * @param out_groups
     * @param out_nodes
     */
    static void loadAll(
            Dataset dataset,
            Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_S,
            std::unordered_map<size_t, std::vector<size_t>> &out_groups,
            std::unordered_map<size_t, std::vector<size_t>> &out_nodes
            );

    static void loadS(
            Dataset dataset,
            Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &S,
            std::unordered_map<size_t, size_t> &out_T,
            std::unordered_map<size_t, size_t> &out_answer);

    template<class Matrix>
    static void saveMatrix(const std::string &filename, const Matrix &mat);

    template<class Matrix>
    static void loadMatrix(const std::string &filename, Matrix &mat);

    template<class SparseMatrix>
    static void saveSparseMatrix(const std::string &filename, const SparseMatrix &mat);

    template<class SparseMatrix>
    static void loadSparseMatrix(const std::string &filename, SparseMatrix &mat);

    static void clean(DatasetLoader::UGraph &graph,
                      std::unordered_map<size_t, std::vector<size_t>> &groups,
                      std::unordered_map<size_t, std::vector<size_t>> &nodes,
                      Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_A);

    static void clean(DatasetLoader::UGraph &graph,
                      std::unordered_map<size_t, std::vector<size_t>> &groups,
                      std::unordered_map<size_t, std::vector<size_t>> &nodes,
                      std::unordered_map<size_t, size_t> &out_T,
                      std::unordered_map<size_t, size_t> &out_answer);
};


#endif //DNE_DATASET_REPO_H
