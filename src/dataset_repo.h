#ifndef DNE_DATASET_REPO_H
#define DNE_DATASET_REPO_H

#include <boost/graph/adjacency_list.hpp>
#include <Eigen/Dense>
#include "dataset_loader.h"

class DatasetRepo {
public:

    enum Dataset{
        YouTube,
        Flickr,
        Karate,
        Wiki
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
            double train_ratio,
            DatasetLoader::UGraph &out_graph,
            std::unordered_map<size_t, size_t> &out_T,
            std::vector<size_t> &out_answer);

    template<class Matrix>
    static void saveMatrix(const std::string &filename, const Matrix &mat);

    template<class Matrix>
    static void loadMatrix(const std::string &filename, Matrix &mat);



    static void clean(double train_ratio,
                      DatasetLoader::UGraph &graph,
                      std::unordered_map<size_t, std::vector<size_t>> &groups,
                      std::unordered_map<size_t, std::vector<size_t>> &nodes,
                      std::unordered_map<size_t, size_t> &out_T,
                      std::vector<size_t> &out_answer);
};


#endif //DNE_DATASET_REPO_H
