add_cuda_executable(tensor FALSE tensor.cc)
target_link_libraries(tensor PUBLIC hshm::cudacxx)

add_cuda_executable(iobench TRUE iobench.cc)
target_link_libraries(iobench PUBLIC hshm::cudacxx MPI::MPI_CXX cufile)

add_cuda_executable(et_tensor TRUE et_tensor.cc)
target_link_libraries(et_tensor PUBLIC hermes::hermes_core_client_gpu eternia_eternia_core MPI::MPI_CXX)
install(TARGETS
    tensor
    iobench
    et_tensor
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)