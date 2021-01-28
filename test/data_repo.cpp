#include <gtest/gtest.h>
#include <boost/graph/graphviz.hpp>

#include "../src/dataset_repo.h"

using namespace boost;

class DataRepo: public ::testing::Test{};

TEST(DataRepo, clean){
    typedef std::pair<int, int> Edge;
    Edge edge_arr[] = {
       Edge(0,1), Edge(0,2), Edge(1,2),
       Edge(2,4), Edge(2,5), Edge(4,5)
    };

    const int num_edges = 6;
    DatasetLoader::UGraph g(edge_arr, edge_arr + num_edges, 6);

    std::unordered_map<size_t, std::vector<size_t>> groups{};
    groups[0] = {0, 2, 3,4};
    groups[1] = {1, 4};

    std::unordered_map<size_t, std::vector<size_t>> nodes{};
    nodes[0] = {0};
    nodes[1] = {1};
    nodes[2] = {0};
    nodes[3] = {0};
    nodes[4] = {0,1};
    nodes[5] = {};

    std::unordered_map<size_t, size_t> out_T;
    std::vector<size_t> out_answer;

    DatasetRepo::clean(0.5, g, groups, nodes, out_T, out_answer);

    EXPECT_EQ(num_vertices(g), 3);
    EXPECT_EQ(num_vertices(g), 3);
    EXPECT_EQ(out_T[0], 0);
    EXPECT_EQ(out_T[1], 1);
    EXPECT_EQ(out_answer[2], 0);
}