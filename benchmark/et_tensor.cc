#include <eternia/transactions/sequential.h>
#include <eternia/vector.h>

__global__ void VectorAdd(et::VectorCtx x_ctx, et::VectorCtx y_ctx,
                          et::VectorCtx z_ctx, size_t N, size_t nthreads) {
  printf("Started vector add!\n");
  size_t size = (N / nthreads);
  size_t off = threadIdx.x * size;
  et::Vector<float> x(x_ctx);
  et::Vector<float> y(y_ctx);
  et::Vector<float> z(z_ctx);
  et::SeqTx a(x, off, size, ET_READ);
  et::SeqTx b(y, off, size, ET_READ);
  et::SeqTx c(z, off, size, ET_WRITE);
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
  printf("Finished vector add!\n");
}
int main() {
  ETERNIA_INIT();
  printf("HERE???\n");
  et::VectorSet<float> x("/bigvec.parquet");
  et::VectorSet<float> y("y"), z("z");
  x.resize(1024);
  y.resize(1024);
  z.resize(1024);
  VectorAdd<<<1, 1>>>(x.Get(0), y.Get(0), z.Get(0), x.size(), 64);
  hshm::GpuApi::Synchronize();
}