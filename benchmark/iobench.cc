#include <cuda_runtime.h>
#include <mpi.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#define USE_CUFILE
#ifdef USE_CUFILE
#include <cufile.h>
#endif

enum class IOPattern { SEQUENTIAL, RANDOM };

enum class IOType { READ, WRITE };

enum class IOEngine { POSIX, CUFILE };

void posix_io(const std::string& filename, size_t transfer_size,
              size_t total_io_size, IOType io_type, IOPattern io_pattern) {
  char* host_buffer;
  char* device_buffer;

  cudaMallocHost(&host_buffer, transfer_size);
  cudaMalloc(&device_buffer, transfer_size);

  if (io_type == IOType::WRITE) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error opening file for writing." << std::endl;
      cudaFreeHost(host_buffer);
      cudaFree(device_buffer);
      return;
    }

    std::vector<char> buffer(transfer_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distrib(
        0, total_io_size - transfer_size);

    size_t num_transfers = total_io_size / transfer_size;
    for (size_t i = 0; i < num_transfers; ++i) {
      size_t offset =
          (io_pattern == IOPattern::RANDOM) ? distrib(gen) : i * transfer_size;
      file.seekp(offset);
      file.write(buffer.data(), transfer_size);
      cudaMemcpy(device_buffer, buffer.data(), transfer_size,
                 cudaMemcpyHostToDevice);
    }
    file.close();
  } else {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error opening file for reading." << std::endl;
      cudaFreeHost(host_buffer);
      cudaFree(device_buffer);
      return;
    }

    std::vector<char> buffer(transfer_size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> distrib(
        0, total_io_size - transfer_size);

    size_t num_transfers = total_io_size / transfer_size;
    for (size_t i = 0; i < num_transfers; ++i) {
      size_t offset =
          (io_pattern == IOPattern::RANDOM) ? distrib(gen) : i * transfer_size;
      file.seekg(offset);
      file.read(buffer.data(), transfer_size);
      cudaMemcpy(device_buffer, buffer.data(), transfer_size,
                 cudaMemcpyHostToDevice);
    }
    file.close();
  }
  cudaFreeHost(host_buffer);
  cudaFree(device_buffer);
}

#ifdef USE_CUFILE
void cufile_io(const std::string& filename, size_t transfer_size,
               size_t total_io_size, IOType io_type, IOPattern io_pattern) {
  CUfileHandle_t fh;
  CUfileDescr_t params;
  memset(&params, 0, sizeof(params));
  params.handle = filename.c_str();
  params.type = CUfileDescrT::CUFILE_DESCRIPTOR_PATHNAME;

  CUfileError_t status = cuFileHandleRegister(&fh, &params);
  if (status != CUFILE_SUCCESS) {
    std::cerr << "cuFileHandleRegister error: " << status << std::endl;
    return;
  }

  char* device_buffer;
  cudaMalloc(&device_buffer, transfer_size);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> distrib(0,
                                                total_io_size - transfer_size);

  if (io_type == IOType::WRITE) {
    size_t num_transfers = total_io_size / transfer_size;
    for (size_t i = 0; i < num_transfers; ++i) {
      size_t offset =
          (io_pattern == IOPattern::RANDOM) ? distrib(gen) : i * transfer_size;
      status = cuFileWrite(device_buffer, fh, transfer_size, offset, 0);
      if (status != CUFILE_SUCCESS) {
        std::cerr << "cuFileWrite error: " << status << std::endl;
        break;
      }
    }
  } else {
    size_t num_transfers = total_io_size / transfer_size;
    for (size_t i = 0; i < num_transfers; ++i) {
      size_t offset =
          (io_pattern == IOPattern::RANDOM) ? distrib(gen) : i * transfer_size;
      status = cuFileRead(device_buffer, fh, transfer_size, offset, 0);
      if (status != CUFILE_SUCCESS) {
        std::cerr << "cuFileRead error: " << status << std::endl;
        break;
      }
    }
  }

  cudaFree(device_buffer);
  cuFileHandleDeregister(fh);
}
#endif

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 7) {
    if (rank == 0) {
      std::cerr
          << "Usage: " << argv[0]
          << " <transfer_size> <total_io_size> <io_pattern (random|sequential)>"
          << " <io_type (read|write)> <io_engine (posix|cufile)> <filename>"
          << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  size_t transfer_size = std::stoull(argv[1]);
  size_t total_io_size = std::stoull(argv[2]);
  std::string io_pattern_str = argv[3];
  std::string io_type_str = argv[4];
  std::string io_engine_str = argv[5];
  std::string filename = argv[6];

  IOPattern io_pattern;
  if (io_pattern_str == "random") {
    io_pattern = IOPattern::RANDOM;
  } else if (io_pattern_str == "sequential") {
    io_pattern = IOPattern::SEQUENTIAL;
  } else {
    if (rank == 0) {
      std::cerr << "Invalid IO pattern: " << io_pattern_str << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  IOType io_type;
  if (io_type_str == "read") {
    io_type = IOType::READ;
  } else if (io_type_str == "write") {
    io_type = IOType::WRITE;
  } else {
    if (rank == 0) {
      std::cerr << "Invalid IO type: " << io_type_str << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  IOEngine io_engine;
  if (io_engine_str == "posix") {
    io_engine = IOEngine::POSIX;
  }
#ifdef USE_CUFILE
  else if (io_engine_str == "cufile") {
    io_engine = IOEngine::CUFILE;
  }
#endif
  else {
    if (rank == 0) {
      std::cerr << "Invalid IO engine: " << io_engine_str << std::endl;
    }
    MPI_Finalize();
    return 1;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto start_time = std::chrono::high_resolution_clock::now();

  switch (io_engine) {
    case IOEngine::POSIX:
      posix_io(filename, transfer_size, total_io_size, io_type, io_pattern);
      break;
#ifdef USE_CUFILE
    case IOEngine::CUFILE:
      cufile_io(filename, transfer_size, total_io_size, io_type, io_pattern);
      break;
#endif
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  long long duration_ms = duration.count();
  long long total_duration;
  MPI_Reduce(&duration_ms, &total_duration, 1, MPI_LONG_LONG, MPI_SUM, 0,
             MPI_COMM_WORLD);

  if (rank == 0) {
    double average_duration = static_cast<double>(total_duration) / size;
    std::cout << "I/O benchmark completed in " << average_duration << " ms"
              << std::endl;
  }

  MPI_Finalize();
  return 0;
}
