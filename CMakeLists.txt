cmake_minimum_required(VERSION 3.16)
project(quadremesh VERSION 1.0.0 LANGUAGES C CXX)

option(QRM_BUILD_PYTHON "Build pybind11 module" ON)
option(QRM_BUILD_TESTS "Build C++ tests" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# Single-threaded, no fast-math: results are bit-for-bit deterministic.
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

set(TP ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(EIGEN_DIR ${TP}/eigen CACHE PATH "Eigen root")
set(LIBIGL_DIR ${TP}/libigl CACHE PATH "libigl root")
find_package(Boost REQUIRED)

# ---------------------------------------------------------------- patched QuadriFlow
set(QF ${TP}/qflow_patched)
add_subdirectory(${QF}/3rd/lemon-1.3.1 ${CMAKE_BINARY_DIR}/lemon EXCLUDE_FROM_ALL)
file(GLOB QF_SRC ${QF}/src/*.cpp)
list(REMOVE_ITEM QF_SRC ${QF}/src/main.cpp ${QF}/src/post-solver.cpp)
add_library(qflow STATIC ${QF_SRC})
target_include_directories(qflow PUBLIC ${QF}/src ${QF}/3rd/pcg32 ${QF}/3rd/pss ${QF}/3rd/lemon-1.3.1
                                        ${CMAKE_BINARY_DIR}/lemon ${EIGEN_DIR} ${Boost_INCLUDE_DIRS})
target_compile_options(qflow PRIVATE -w -include ${QF}/src/qflow_quiet.hpp)
target_link_libraries(qflow PUBLIC lemon)

# ---------------------------------------------------------------- Instant Meshes core (fallback extractor)
set(IM ${TP}/instant-meshes)
set(TBB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(TBB_BUILD_TBBMALLOC OFF CACHE BOOL "" FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL "" FORCE)
set(TBB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TBB_INSTALL_TARGETS OFF CACHE BOOL "" FORCE)
add_subdirectory(${IM}/ext/tbb ${CMAKE_BINARY_DIR}/tbb EXCLUDE_FROM_ALL)
set_target_properties(tbb_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(imcore STATIC
  ${IM}/src/adjacency.cpp ${IM}/src/bvh.cpp ${IM}/src/dedge.cpp ${IM}/src/extract.cpp ${IM}/src/field.cpp
  ${IM}/src/hierarchy.cpp ${IM}/src/meshstats.cpp ${IM}/src/normal.cpp ${IM}/src/subdivide.cpp
  ${IM}/src/serializer.cpp ${IM}/src/smoothcurve.cpp ${IM}/src/cleanup.cpp ${IM}/src/reorder.cpp
  ${IM}/src/meshio.cpp ${IM}/ext/rply/rply.c)
target_include_directories(imcore PUBLIC ${IM}/src ${IM}/ext/pcg32 ${IM}/ext/dset ${IM}/ext/pss ${IM}/ext/tbb/include
                                         ${IM}/ext/half ${IM}/ext/rply ${EIGEN_DIR})
target_compile_options(imcore PRIVATE -w $<$<COMPILE_LANGUAGE:CXX>:-include ${IM}/im_compat.h>)
target_link_libraries(imcore PUBLIC tbb_static pthread)

# ---------------------------------------------------------------- core
add_library(quadremesh_core STATIC
  src/io.cpp src/geometry.cpp src/preprocess.cpp src/features.cpp src/field.cpp
  src/qflow_driver.cpp src/im_driver.cpp src/postprocess.cpp src/transfer.cpp src/metrics.cpp src/remesh.cpp)
target_include_directories(quadremesh_core PUBLIC include ${EIGEN_DIR} PRIVATE src ${LIBIGL_DIR}/include ${TP}/cgltf)
target_link_libraries(quadremesh_core PUBLIC qflow imcore)
target_compile_options(quadremesh_core PRIVATE -Wall -Wno-sign-compare -Wno-unused-variable -Wno-int-in-bool-context -Wno-maybe-uninitialized)

add_executable(quadremesh apps/quadremesh_cli.cpp)
target_link_libraries(quadremesh PRIVATE quadremesh_core)
add_executable(quadeval apps/quadeval_cli.cpp)
target_link_libraries(quadeval PRIVATE quadremesh_core)

# ---------------------------------------------------------------- python
if(QRM_BUILD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  add_subdirectory(${TP}/pybind11 ${CMAKE_BINARY_DIR}/pybind11)
  pybind11_add_module(_quadremesh python/bindings.cpp)
  target_link_libraries(_quadremesh PRIVATE quadremesh_core)
  target_include_directories(_quadremesh PRIVATE src)
endif()

# ---------------------------------------------------------------- tests
if(QRM_BUILD_TESTS)
  enable_testing()
  add_executable(test_core tests/test_core.cpp)
  target_link_libraries(test_core PRIVATE quadremesh_core)
  target_include_directories(test_core PRIVATE src ${LIBIGL_DIR}/include)
  add_test(NAME core COMMAND test_core ${CMAKE_CURRENT_SOURCE_DIR}/data/models)
endif()
