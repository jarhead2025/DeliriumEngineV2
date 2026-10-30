// FILEPATH: src/math/CMakeLists.txt

find_package(Eigen3 REQUIRED)

add_library(math_lib STATIC 
  cholesky_decompose.cpp
  cholesky_eigen.cpp
)

target_include_directories(math_lib PRIVATE ${Eigen3_INCLUDE_DIRS})
target_link_libraries(math_lib Eigen::Eigen)
