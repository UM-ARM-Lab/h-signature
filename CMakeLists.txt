project(h_signature)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)

find_package (Eigen3 REQUIRED NO_MODULE)
include(CTest)

include_directories("include")
add_library(h_signature STATIC src/h_signature.cpp)
target_link_libraries(h_signature PUBLIC Eigen3::Eigen)

add_executable(test_h_signature test/test_h_signature.cpp)
target_link_libraries(test_h_signature PRIVATE h_signature)

add_test(NAME TestHSignature COMMAND test_h_signature)
