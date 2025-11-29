cmake_minimum_required(VERSION 3.16)
project(UR5eSim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 1. Find the libraries automatically
# This searches for MuJoCo's config files
find_package(mujoco REQUIRED)
# This finds the system-installed GLFW and OpenGL
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# 2. Define the executable
add_executable(ur5e_sim_tutorial src/ur5e_sim_tutorial.cpp)
add_executable(ur5e_sim_FL src/ur5e_sim_FL.cpp)

# 3. Link the libraries
# The 'mujoco::mujoco' target provides its own include/link paths
target_link_libraries(ur5e_sim_tutorial PRIVATE
    mujoco::mujoco  # This imported target adds -I, -L, and -lmujoco
    glfw            # This imported target adds -lglfw
    OpenGL::GL      # This imported target adds -lGL
    m               # Math library still needs to be added
)

target_link_libraries(ur5e_sim_FL PRIVATE
    mujoco::mujoco  # This imported target adds -I, -L, and -lmujoco
    glfw            # This imported target adds -lglfw
    OpenGL::GL      # This imported target adds -lGL
    m               # Math library still needs to be added
    Eigen3::Eigen
)