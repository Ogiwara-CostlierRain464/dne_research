cmake_minimum_required(VERSION 3.13)

set(ENV{CC} gcc)
set(ENV{CXX} g++)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

project(DNE)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE PROJECT_SOURCES src/*.cpp)
file(GLOB_RECURSE PROJECT_HEADERS src/*.h)

find_package(Boost 1.40
    COMPONENTS graph REQUIRED
)


add_executable(${PROJECT_NAME}
        ${PROJECT_SOURCES}
        ${PROJECT_HEADERS}
)

include_directories(
        ${Boost_INCLUDE_DIR}
        third_party/eigen
)

target_link_libraries(
        ${PROJECT_NAME}
        LINK_PUBLIC
        ${Boost_LIBRARIES}
)