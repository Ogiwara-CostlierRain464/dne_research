cmake_minimum_required(VERSION 3.13)

#set(ENV{CC} gcc)
#set(ENV{CXX} g++)

set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")


set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -march=native")

project(DNE)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE PROJECT_SOURCES src/*.cpp)
file(GLOB_RECURSE PROJECT_HEADERS src/*.h)

find_package(Boost 1.40 COMPONENTS serialization graph REQUIRED)
add_subdirectory(third_party/glog)
add_subdirectory(third_party/gflags)


add_executable(${PROJECT_NAME}
        ${PROJECT_SOURCES}
        ${PROJECT_HEADERS}
)

include_directories(
        ${Boost_INCLUDE_DIR}
        third_party/eigen
        third_party/json/include
)

target_link_libraries(
        ${PROJECT_NAME}
        LINK_PUBLIC
        ${Boost_LIBRARIES}
        gflags
        glog
)

add_subdirectory(third_party/json)
add_subdirectory(third_party/libpopcnt)


enable_testing()


# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
        RESULT_VARIABLE result
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
    message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
        ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
        EXCLUDE_FROM_ALL)

# The gtest/gtest_main targets carry header search path
# dependencies automatically when using CMake 2.8.11 or
# later. Otherwise we have to add them here ourselves.
if (CMAKE_VERSION VERSION_LESS 2.8.11)
    include_directories("${gtest_SOURCE_DIR}/include")
endif()

file(GLOB_RECURSE TEST_SOURCES test/*.cpp)
file(GLOB_RECURSE TEST_HEADERS test/*.h)

# main.cppのみ外す
list(REMOVE_ITEM PROJECT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)


add_executable(tests
        ${TEST_SOURCES}
        ${TEST_HEADERS}
        ${PROJECT_SOURCES}
        ${PROJECT_HEADERS}
)
target_link_libraries(tests
        gtest_main
        gflags
        glog
        ${Boost_LIBRARIES}
)
add_test(NAME example_test COMMAND tests)
