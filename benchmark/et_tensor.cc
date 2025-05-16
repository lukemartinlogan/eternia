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

__global__ void HermesPut(et::VectorCtx x_ctx, size_t size, int count) {
  printf("Started hermes put kernel: %llu!\n",
         hshm::GpuApi::GetGlobalThreadId());
  et::Vector<float> x(x_ctx);
  auto *main_alloc = CHI_CLIENT->main_alloc_;
  hipc::MemContext mctx;
  mctx.tid_ = hshm::ThreadId(hshm::GpuApi::GetGlobalThreadId());
  main_alloc->CreateTls(mctx);
  hipc::CtxAllocator<CHI_MAIN_ALLOC_T> ctx_alloc(mctx, main_alloc);
  hermes::Context ctx;
  ctx.mctx_ = ctx_alloc.ctx_;
  hermes::Bucket bkt(x.bkt_.id_, x.bkt_.mdm_, ctx);
  hermes::Blob blob(size);
  chi::string blob_name = MakeBlobName(ctx_alloc, ctx.mctx_.tid_.tid_);
  for (int i = 0; i < count; ++i) {
    bkt.Put(blob_name, blob, ctx);
  }
}

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

__global__ void VectorAddEmu(float *data, size_t size, size_t nthreads) {
  size_t size_per_thread = size / nthreads;
  size_t tid = hshm::GpuApi::GetGlobalThreadId();
  size_t off = size_per_thread * tid;
  for (int j = 0; j < 16; ++j) {
    for (size_t i = 0; i < size_per_thread; ++i) {
      data[i + off] = i * 2 + off + 10;
    }
  }
}

void TestHermesPut() {
  HERMES_INIT();
  et::VectorSet<float> x("/bigvec.parquet");
  x.resize(1024);
  hshm::Timer t;
  t.Resume();
  HermesPut<<<16, 32>>>(x.Get(0), MEGABYTES(1), 1);
  hshm::GpuApi::Synchronize();
  t.Pause();
  printf("TOTAL TIME: %lf msec", t.GetMsec());
}

void TestVectorAddEmu(size_t block, size_t warp, size_t size) {
  printf("Vector add emu with block=%llu warp=%llu and size=%llu\n", block,
         warp, size);
  hshm::Timer t;
  t.Resume();
  float *data = hshm::GpuApi::Malloc<float>(size);
  VectorAddEmu<<<block, warp>>>(data, size, block * warp);
  hshm::GpuApi::Synchronize();
  t.Pause();
  printf("TOTAL TIME: %lf msec", t.GetMsec());
}

void TestEterniaSequential() {
  ETERNIA_INIT();
  et::VectorSet<float> x("/bigvec.parquet");
  et::VectorSet<float> y("y"), z("z");
  x.resize(1024);
  y.resize(1024);
  z.resize(1024);
  VectorAdd<<<1, 1>>>(x.Get(0), y.Get(0), z.Get(0), x.size(), 64);
  hshm::GpuApi::Synchronize();
}

int main(int argc, char **argv) {
  size_t block = 0;
  size_t warp = 0;
  size_t size = 0;

  if (argc >= 4) {
    block = atoi(argv[1]);
    warp = atoi(argv[2]);
    size = hshm::ConfigParse::ParseSize(argv[3]);
  }
  TestVectorAddEmu(block, warp, size);
}
