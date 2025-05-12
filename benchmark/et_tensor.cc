#include <eternia/vector.h>

__global__ void vectorAdd(et::Vector<float> v[3], int N, et::Comm grid) {
  int size = (N / grid.x);
  int off = threadIdx.x * size;
  et::SeqTx a(v[0], off, size, ET_RONLY);
  et::SeqTx b(v[1], off, size, ET_RONLY);
  et::SeqTx c(v[2], off, size, ET_WONLY);
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}
int main() {
  et::VectorSet<float> x("/bigvec.parquet");
  et::VectorSet<float> y(x.size()), z(x.size());
  VectorAdd<<<1, 64>>>({x, y, z}, x.size());
}