cmake_minimum_required(VERSION 3.10)
project(diff_sim_core VERSION 25.1 LANGUAGES CXX)

include_directories(include)

set(CMAKE_CXX_STANDARD 17)
# Optimization flags (essential for physics later)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# 1. Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# 2. Find Pybind11
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmake
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 REQUIRED)

# 3. Find Eigen (We will need it soon)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 4. Define the Module "forgeNN_cpp"
# We only have one source file for now.
# C++ Source Files
set(SOURCES
    src/bindings.cpp
    src/core.cpp
    src/activations.cpp
    src/optimizers.cpp
)
pybind11_add_module(forgeNN_cpp ${SOURCES})

# Link Eigen
target_link_libraries(forgeNN_cpp PRIVATE Eigen3::Eigen)

# Statically link libstdc++ to avoid GLIBCXX version mismatch with Conda
target_link_options(forgeNN_cpp PRIVATE -static-libstdc++ -static-libgcc)