cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(dcgan)

find_package(Torch REQUIRED)

# DCGAN training
# add_executable(dcgan dcgan.cpp)
# target_link_libraries(dcgan "${TORCH_LIBRARIES}")
# set_property(TARGET dcgan PROPERTY CXX_STANDARD 14)

# Load pytorch model by TorchScript and do the test
add_executable(resnet_speed_test resnet_speed_test.cpp)
target_link_libraries(resnet_speed_test "${TORCH_LIBRARIES}")
set_property(TARGET resnet_speed_test PROPERTY CXX_STANDARD 14)