add_cuda_library(libgpu STATIC FALSE
    ${CMAKE_CURRENT_SOURCE_DIR}/libgpu.cc)
target_link_libraries(libgpu PUBLIC hermes::hermes_core_client_gpu)