#include <eternia/transactions/sequential.h>
#include <eternia/vector.h>

/** Make blob name */
HSHM_GPU_FUN
chi::string MakeBlobName(hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &alloc,
                         size_t page_id) {
  chi::string blob_name(alloc, sizeof(size_t));
  memcpy(blob_name.data(), &page_id, sizeof(size_t));
  return blob_name;
}

__global__ void HermesPut(et::VectorCtx x_ctx, size_t N, size_t nthreads) {
  printf("Started vector add!\n");
  size_t size = (N / nthreads);
  size_t off = threadIdx.x * size;
  et::Vector<float> x(x_ctx);
  auto *main_alloc = CHI_CLIENT->main_alloc_;
  hipc::MemContext mctx;
  mctx.tid_ = hshm::ThreadId(hshm::GpuApi::GetGlobalThreadId());
  main_alloc->CreateTls(mctx);
  hipc::CtxAllocator<CHI_MAIN_ALLOC_T> ctx_alloc(mctx, main_alloc);
  printf("Scoped TLS aquired!\n");
  hermes::Context ctx;
  ctx.mctx_ = ctx_alloc.ctx_;
  hermes::Bucket bkt(x.bkt_.id_, x.bkt_.mdm_, ctx);
  printf("Bucket made!\n");
  hermes::Blob blob(MEGABYTES(1));
  printf("Creating blob!\n");
  chi::string blob_name = MakeBlobName(ctx_alloc, off);
  printf("Made blob name!\n");
  if (hshm::GpuApi::GetGlobalThreadId() == 0) {
    for (int i = 0; i < 1; ++i) {
      bkt.Put(blob_name, blob, ctx);
    }
  }
  printf("Finished vector add!\n");
}

__global__ void VectorAdd(et::VectorCtx x_ctx, et::VectorCtx y_ctx,
                          et::VectorCtx z_ctx, size_t N, size_t nthreads) {
  printf("Started vector add!\n");
  size_t size = (N / nthreads);
  size_t off = threadIdx.x * size;
  et::Vector<float> x(x_ctx);
  // et::Vector<float> y(y_ctx);
  // et::Vector<float> z(z_ctx);
  // et::SeqTx a(x, off, size, ET_READ);
  // et::SeqTx b(y, off, size, ET_READ);
  // et::SeqTx c(z, off, size, ET_WRITE);
  // for (int i = 0; i < size; ++i) {
  //   c[i] = a[i] + b[i];
  // }
  printf("Finished vector add!\n");
}

int main() {
  HERMES_INIT();
  printf("HERE???\n");
  et::VectorSet<float> x("/bigvec.parquet");
  x.resize(1024);
  hshm::Timer t;
  t.Resume();
  // auto *mux = hshm::GpuApi::MallocManaged<Mutex>(sizeof(Mutex));
  // hipc::Allocator::ConstructObj(*mux);
  HermesPut<<<2, 32>>>(x.Get(0), x.size(), 64);
  // TestMutex<<<256, 256>>>(mux);
  hshm::GpuApi::Synchronize();
  t.Pause();
  printf("TOTAL TIME: %lf msec", t.GetMsec());
}

// int main() {
//   ETERNIA_INIT();
//   printf("HERE???\n");
//   et::VectorSet<float> x("/bigvec.parquet");
//   et::VectorSet<float> y("y"), z("z");
//   x.resize(1024);
//   y.resize(1024);
//   z.resize(1024);
//   VectorAdd<<<1, 1>>>(x.Get(0), y.Get(0), z.Get(0), x.size(), 64);
//   hshm::GpuApi::Synchronize();
// }
