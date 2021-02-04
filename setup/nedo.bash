# Nedoのマシン上でcmakeするときのコード

# shellcheck disable=SC2164
cd cmake-build-release
cmake .. cmake -DBOOST_ROOT=/home/albicilla/boost_1_69_0/stage/ -DBOOST_LIBRARYDIR=/home/albicilla/boost_1_69_0_stage/lib   ..