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
        BlogCatalog,
        DBLP
    };

    static void load(
            Dataset dataset,
            double train_ratio,
            Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_S,
            Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_L,
            std::unordered_map<size_t, size_t> &out_T,
            std::vector<size_t> &out_answer,
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
            Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_L,
            std::vector<std::vector<size_t>> &out_groups, // {group_id: [ node_ids ]}
            std::vector<size_t> &out_nodes // {node_id:  group_id }
            );


    template<class SparseMatrix>
    static void saveSparseMatrix(const std::string &filename, const SparseMatrix &mat);

    template<class SparseMatrix>
    static void loadSparseMatrix(const std::string &filename, SparseMatrix &mat);

    static void clean(DatasetLoader::UGraph &graph,
                      std::unordered_map<size_t, std::vector<size_t>> &dirty_groups,
                      std::unordered_map<size_t, std::vector<size_t>> &dirty_nodes,
                      std::vector<std::vector<size_t>> &out_groups,
                      std::vector<size_t> &out_nodes,
                      Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_A,
                      Eigen::SparseMatrix<double, 0, std::ptrdiff_t> &out_L);
};


#endif //DNE_DATASET_REPO_H
