add_cuda_executable(tensor FALSE tensor.cc)
target_link_libraries(tensor PUBLIC hshm::cudacxx)

add_cuda_executable(iobench FALSE iobench.cc)
target_link_libraries(iobench PUBLIC hshm::cudacxx MPI::MPI_CXX cufile)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
add_cuda_executable(et_tensor FALSE et_tensor.cc)
target_link_libraries(et_tensor PUBLIC eternia_eternia_core_client_gpu MPI::MPI_CXX)
install(TARGETS
    tensor
    iobench
    et_tensor
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)