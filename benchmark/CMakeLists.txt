add_cuda_executable(tensor FALSE tensor.cc)
target_link_libraries(tensor PUBLIC hshm::cudacxx)

add_cuda_executable(iobench FALSE iobench.cc)
target_link_libraries(iobench PUBLIC hshm::cudacxx MPI::MPI_CXX cufile)

install(TARGETS
    tensor
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)