#include <eternia/transactions/sequential.h>
#include <eternia/vector.h>

__global__ void vectorAdd(et::Vector<float> v[3], size_t N, size_t x) {
  size_t size = (N / x);
  size_t off = threadIdx.x * size;
  et::SeqTx a(v[0], off, size, ET_READ);
  et::SeqTx b(v[1], off, size, ET_READ);
  et::SeqTx c(v[2], off, size, ET_WRITE);
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}
int main() {
  et::VectorSet<float> x("/bigvec.parquet");
  et::VectorSet<float> y(x.size()), z(x.size());
  VectorAdd<<<1, 64>>>({x, y, z}, x.size());
}