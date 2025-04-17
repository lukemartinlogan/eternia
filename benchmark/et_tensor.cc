#include <eternia/vector.h>
__global__ void vectorAdd(et::Vector<float> v[3],
                          int N, et::Comm grid) {
  int size = (N / grid.x);
  int off = threadIdx.x * size;
  et::Comm comm(1, 16);
  et::SeqTx a(v[0], off, size, ET_RONLY, comm);
  et::SeqTx b(v[1], off, size, ET_RONLY, comm);
  et::SeqTx c(v[2], off, size, ET_WONLY, comm);
  for (int i = 0; i < size; ++i) {
    c[i] = a[i] + b[i];
  }
}
int main() {
  et::Vector<float> x("/bigvec.parquet");
  et::vector<float> y(x.size()), z(x.size());
  et::Comm grid(1, 64);
  VectorAdd<<<1, 64>>>(
      {x, y, z}, x.size(), grid);
}