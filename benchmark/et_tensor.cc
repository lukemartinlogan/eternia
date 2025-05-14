#include <eternia/transactions/sequential.h>
#include <eternia/vector.h>

__global__ void VectorAdd(et::Vector<float> v[3], size_t N, size_t nthreads) {
  size_t size = (N / nthreads);
  size_t off = threadIdx.x * size;
  et::SeqTx a(v[0], off, size, ET_READ);
  et::SeqTx b(v[1], off, size, ET_READ);
  et::SeqTx c(v[2], off, size, ET_WRITE);
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}
int main() {
  ETERNIA_INIT();
  printf("HERE???\n");
  et::VectorSet<float> x("/bigvec.parquet");
  et::VectorSet<float> y("y"), z("z");
  x.resize(1024);
  y.resize(1024);
  // et::Vector<float> vs[] = {x.Get(0), y.Get(0), z.Get(0)};
  // VectorAdd<<<1, 64>>>(vs, x.size(), 64);
}