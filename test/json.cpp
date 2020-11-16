#include <gtest/gtest.h>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <nlohmann/json.hpp>

using namespace std;
using json = nlohmann::json;

class JSON: public ::testing::Test{};

const string LARGE_JSON = "/Users/ogiwara/Downloads/dblp.v12.json";



TEST(JSON, load){



  bool read_id_flag = false;
  bool read_ref_flag = false;
  bool read_keyword_flag = false;


  std::vector<uint> ids;
  std::vector<uint> refs;


  auto x = [&](int depth, json::parse_event_t event, json& parsed){

    if(1 + 1 == 2){

    }

    if(event == json::parse_event_t::array_start){
      if(read_keyword_flag or read_ref_flag){
        return true;
      }
    }

    if(event == json::parse_event_t::value){

      if(read_id_flag){
        auto r = parsed.get_ref<json::number_unsigned_t&>();
        ids.push_back(r);
        read_id_flag = false;
        return true;
      }
      if(read_ref_flag){
        auto r = parsed.get_ref<json::number_unsigned_t&>();
        refs.push_back(r);
        read_ref_flag = false;
        return true;
      }
      if(read_keyword_flag){

        read_keyword_flag = false;
        return true;
      }
    }

    if(event == json::parse_event_t::key){
      auto r = parsed.get_ref<string&>();
      cout << r << endl;
      if(depth == 2){
        if(r == "id"){
          read_id_flag = true;
          return true;
        }

        if(r == "references"){
          read_ref_flag = true;
          return true;
        }

      }
      if(r == "keywords"){
        read_keyword_flag = true;
        return true;
      }
      if(r == "fos.name"){
        return true;
      }
    }

    // マルチラベルへの対応はまず必要
    // で、keywordはどうする？例えばDatabaseとDatabase transactionは区別しない方がいいし
    // ここは相談すべきだね

    return false;
  };

  std::ifstream i(LARGE_JSON);
  json::parse(i, x);

  //read_json("/Users/ogiwara/Downloads/dblp.v12.json", pt);
//  EXPECT_EQ(pt.get_optional<int>("Data.value"), 3);
}

struct A{
  int id;
  int k;
};

namespace ns {
  // a simple struct to model a person
  struct person {
    std::string name;
    std::string address;
    int age;
  };

  void to_json(json& j, const person& p) {
    j = json{{"name", p.name}, {"address", p.address}, {"age", p.age}};
  }

  void from_json(const json& j, person& p) {
    j.at("name").get_to(p.name);
    j.at("address").get_to(p.address);
    j.at("age").get_to(p.age);
  }
}

TEST(JSON, write){

  ns::person p {"Ned Flanders", "744 Evergreen Terrace", 60};

  json j = p;
  std::ofstream o("p.json");
  o << j;

  // https://www.aminer.org/citation
  // https://academic.microsoft.com/topics/77088390,75949130?fullPath=false
  // https://nlohmann.github.io/json/api/basic_json/get_ref/


}