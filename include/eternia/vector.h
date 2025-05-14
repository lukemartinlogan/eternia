#ifndef ETERNIA_VECTOR_H
#define ETERNIA_VECTOR_H

#include <hermes/hermes.h>
#include <hermes_shm/constants/macros.h>

#include "eternia_core/eternia_core_client.h"
#include "libgpu.h"

namespace eternia {

#define DEFAULT_TCACHE_PAGE_SIZE 4096
#define DEFAULT_TCACHE_SLOTS 256
#define TCACHE_MIN_SCORE 0.8

struct Page {
  size_t id_;
  float score_;
  Page *next_;
  char buf_[];
};

struct Context {
  size_t tcache_page_size_ = DEFAULT_TCACHE_PAGE_SIZE;
  size_t tcache_page_slots_ = DEFAULT_TCACHE_SLOTS;
  size_t gcache_page_size_ = MEGABYTES(1);
};

struct PageAllocator {
  size_t head_ = 0, tail_ = 0;
  size_t DEPTH;
  size_t PAGE_SIZE;
  char *page_data_ = nullptr;
  Page **alloc_ = nullptr;

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  PageAllocator() = default;

  /** Destructor */
  HSHM_INLINE_CROSS_FUN
  ~PageAllocator() {
    if (page_data_) {
      free(page_data_);
    }
    if (alloc_) {
      free(alloc_);
    }
  }

  /** Create page allocator (per-kernel) */
  HSHM_GPU_FUN
  PageAllocator(const Context &ctx) {
    PAGE_SIZE = sizeof(Page) + ctx.tcache_page_size_;
    size_t size = PAGE_SIZE * ctx.tcache_page_slots_;
    DEPTH = ctx.tcache_page_slots_;
    page_data_ = (char *)malloc(size);
    memset(page_data_, 0, size);
    alloc_ = (Page **)malloc(ctx.tcache_page_slots_ * sizeof(Page *));
  }

  /** Allocate a page */
  HSHM_INLINE_CROSS_FUN Page *Allocate() {
    if (tail_ - head_ >= DEPTH) {
      return nullptr;
    } else if (tail_ < DEPTH) {
      size_t tail = tail_++;
      return (Page *)(page_data_ + (tail * PAGE_SIZE));
    } else {
      size_t tail = (tail_++) % DEPTH;
      return alloc_[tail];
    }
  }

  /** Free a page */
  HSHM_INLINE_CROSS_FUN
  void Free(Page *page) {
    size_t head = (head_++) % DEPTH;
    alloc_[head] = page;
  }
};

struct PageMap {
  size_t DEPTH;
  Page **bkts_ = nullptr;

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  PageMap() {}

  /** Destructor */
  HSHM_INLINE_CROSS_FUN
  ~PageMap() {
    if (bkts_) {
      free(bkts_);
    }
  }

  /** GPU constructor */
  HSHM_GPU_FUN
  PageMap(const Context &ctx) {
    DEPTH = ctx.tcache_page_slots_;
    size_t size = ctx.tcache_page_slots_ * sizeof(Page *);
    bkts_ = (Page **)malloc(size);
    memset(bkts_, 0, size);
  }

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN
  PageMap(const PageMap &other) {
    DEPTH = other.DEPTH;
    bkts_ = other.bkts_;
  }

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN
  PageMap(PageMap &&other) noexcept {
    DEPTH = other.DEPTH;
    bkts_ = other.bkts_;
    other.bkts_ = nullptr;
  }

  /** Copy assignment */
  HSHM_INLINE_CROSS_FUN
  PageMap &operator=(const PageMap &other) {
    if (this != &other) {
      DEPTH = other.DEPTH;
      bkts_ = other.bkts_;
    }
    return *this;
  }

  /** Move assignment */
  HSHM_INLINE_CROSS_FUN
  PageMap &operator=(PageMap &&other) noexcept {
    if (this != &other) {
      DEPTH = other.DEPTH;
      bkts_ = other.bkts_;
      other.bkts_ = nullptr;
    }
    return *this;
  }

  /** Emplace page into map */
  HSHM_INLINE_CROSS_FUN
  void Emplace(const PageRegion &page_id, Page *page) {
    size_t bkt_id = page_id.Hash() % DEPTH;
    Page *bkt = bkts_[bkt_id];
    page->next_ = bkt;
    bkts_[page_id.Hash() % DEPTH] = page;
  }

  /** Find page in map */
  HSHM_INLINE_CROSS_FUN
  Page *Find(const PageRegion &page_id) {
    size_t bkt_id = page_id.Hash() % DEPTH;
    Page *bkt = bkts_[bkt_id];
    while (bkt) {
      if (bkt->id_ == page_id.id_) {
        return bkt;
      }
      bkt = bkt->next_;
    }
    return nullptr;
  }

  /** Remove page from map */
  HSHM_INLINE_CROSS_FUN
  Page *Remove(const PageRegion &page_id) {
    size_t bkt_id = page_id.Hash() % DEPTH;
    Page *bkt = bkts_[bkt_id];
    if (bkt == nullptr) {
      return nullptr;
    }

    if (bkt->id_ == page_id.id_) {
      bkts_[page_id.Hash() % DEPTH] = bkt->next_;
      bkt->next_ = nullptr;
      return bkt;
    }

    Page *prev = bkt;
    bkt = bkt->next_;
    while (bkt) {
      if (bkt->id_ == page_id.id_) {
        prev->next_ = bkt->next_;
        bkt->next_ = nullptr;
        return bkt;
      }
      prev = bkt;
      bkt = bkt->next_;
    }
    return nullptr;
  }
};

template <typename T>
class Vector {
 public:
  typedef T Type;

 public:
  hermes::Bucket bkt_;
  Context ctx_;
  CHI_DATA_GPU_ALLOC_T *gpu_alloc_;
  PageAllocator page_alloc_;
  PageMap page_map_;
  Page *last_page_;
  hipc::FullPtr<GpuCache> gcache_;
  size_t page_bnds_ = 0;  // A counter for number of page boundaries crossed
  size_t size_ = 0;       // Size of the vector

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Vector() = default;

  /** Emplace constructor */
  HSHM_INLINE_CROSS_FUN
  Vector(hermes::Bucket bkt, const Context &ctx, int gpu_id) {
    bkt_ = bkt;
    ctx_ = ctx;
    gpu_alloc_ = CHI_CLIENT->GetGpuDataAlloc(gpu_id);
    gcache_ = ETERNIA_CLIENT->gcache_[gpu_id];
    last_page_ = nullptr;
  }

  /** Initialize vector on GPU thread */
  HSHM_GPU_FUN
  Vector &LocalInit() {
    page_alloc_ = PageAllocator(ctx_);
    page_map_ = PageMap(ctx_);
    return *this;
  }

  /** Copy constructor */
  HSHM_INLINE_CROSS_FUN
  Vector(const Vector &other) {
    bkt_ = other.bkt_;
    ctx_ = other.ctx_;
    gpu_alloc_ = other.gpu_alloc_;
    page_alloc_ = other.page_alloc_;
    page_map_ = other.page_map_;
    last_page_ = other.last_page_;
    gcache_ = other.gcache_;
    page_bnds_ = other.page_bnds_;
    size_ = other.size_;
  }

  /** Move constructor */
  HSHM_INLINE_CROSS_FUN
  Vector(Vector &&other) noexcept {
    bkt_ = std::move(other.bkt_);
    ctx_ = std::move(other.ctx_);
    gpu_alloc_ = other.gpu_alloc_;
    page_alloc_ = std::move(other.page_alloc_);
    page_map_ = std::move(other.page_map_);
    last_page_ = other.last_page_;
    gcache_ = std::move(other.gcache_);
    page_bnds_ = other.page_bnds_;
    size_ = other.size_;
    other.last_page_ = nullptr;
    other.size_ = 0;
  }

  /** Copy assignment */
  HSHM_INLINE_CROSS_FUN
  Vector &operator=(const Vector &other) {
    if (this != &other) {
      bkt_ = other.bkt_;
      ctx_ = other.ctx_;
      gpu_alloc_ = other.gpu_alloc_;
      page_alloc_ = other.page_alloc_;
      page_map_ = other.page_map_;
      last_page_ = other.last_page_;
      gcache_ = other.gcache_;
      page_bnds_ = other.page_bnds_;
      size_ = other.size_;
    }
    return *this;
  }

  /** Move assignment */
  HSHM_INLINE_CROSS_FUN
  Vector &operator=(Vector &&other) noexcept {
    if (this != &other) {
      bkt_ = std::move(other.bkt_);
      ctx_ = std::move(other.ctx_);
      gpu_alloc_ = other.gpu_alloc_;
      page_alloc_ = std::move(other.page_alloc_);
      page_map_ = std::move(other.page_map_);
      last_page_ = other.last_page_;
      gcache_ = std::move(other.gcache_);
      page_bnds_ = other.page_bnds_;
      size_ = other.size_;
      other.last_page_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  /** Get size */
  size_t size() const { return size_; }

  /** Locally rescore a tcache page */
  HSHM_INLINE_GPU_FUN
  void SuggestScore(const PageRegion &region, float score) {
    Page *page = page_map_.Find(region);
    if (page) {
      page->score_ = score;
      if (score < TCACHE_MIN_SCORE && region.modified_) {
        FlushPage(region);
      }
    }
  }

  /** Locally rescore a tcache page */
  HSHM_INLINE_GPU_FUN
  void SolidifyScore(const PageRegion &region, float score) {
    Page *page = page_map_.Find(region);
    // Prioritize previously suggested page score
    if (page && score < page->score_) {
      return;
    }
    // Fault into tcache if score is high enough
    if (!page && score >= TCACHE_MIN_SCORE) {
      page = FaultPage(region, score);
    }
    // Evict from tcache if score is too low
    if (page && page->score_ < TCACHE_MIN_SCORE) {
      InvalidatePage(region);
    }
    // Rescore the page in gcache
    MemTask mem_task;
    mem_task.op_ = GcacheOp::kRescore;
    mem_task.tag_id_ = bkt_.id_;
    mem_task.region_ = region.ChangePageSize(ctx_.gcache_page_size_);
    mem_task.score_ = score;
    SubmitMemTask(mem_task);
  }

  /** Submit MemTask to Gcache */
  HSHM_INLINE_GPU_FUN
  void SubmitMemTask(const MemTask &task) { gcache_->SubmitMemTask(task); }

  /** Find value at offset off in tcache.
   * @param off Offset is in units of T.
   * @param val Pointer to value at offset off.
   */
  HSHM_INLINE_GPU_FUN
  bool FindValInTcache(size_t off, T *&val) {
    PageRegion page_id(off * sizeof(T), ctx_.tcache_page_size_);
    if (!FindPageInTcache(off, last_page_, page_id)) {
      return false;
    }
    val = GetValFromPage(page_id);
    return true;
  }

 private:
  /** Find page containing offset off in tcache, given PageRegion
   * @param off Offset is in units of T.
   * @param page Pointer to page containing offset off.
   * @param page_id PageRegion of the page to find.
   * @return True if the page is found in tcache, false otherwise.
   */
  HSHM_INLINE_GPU_FUN
  bool FindPageInTcache(size_t off, Page *&page, const PageRegion &page_id) {
    // Check the last pointer
    if (last_page_ && last_page_->id_ == page_id.id_) {
      return true;
    }
    // Check the hash pointer
    page_bnds_++;
    page = page_map_.Find(page_id);
    return page != nullptr;
  }

  /** Gets the value in page using PageRegion */
  HSHM_INLINE_GPU_FUN
  T *GetValFromPage(const PageRegion &page_id) {
    if (last_page_) {
      return (T *)(last_page_->buf_)[page_id.off_];
    }
    return nullptr;
  }

  /** Allocate a tcache page */
  HSHM_INLINE_GPU_FUN
  Page *AllocatePage(size_t off) {
    PageRegion page_id(off, ctx_.tcache_page_size_);
    Page *page = page_alloc_.Allocate();
    if (!page) {
      return nullptr;
    }
    page_map_.Emplace(page_id, page);
    return page;
  }

  /** Free page from tcache */
  HSHM_INLINE_GPU_FUN
  void FreePage(size_t off) {
    PageRegion page_id(off, ctx_.tcache_page_size_);
    Page *page = page_map_.Remove(page_id);
    if (page == last_page_) {
      last_page_ = nullptr;
    }
  }

  /** Fault page from gcache into tcache */
  HSHM_INLINE_GPU_FUN
  Page *FaultPage(const PageRegion &page_id, float score) {
    // Make sure page is not already in tcache
    Page *page = page_map_.Find(page_id);
    if (page) {
      return page;
    }
    // Create page in tcache
    page = page_alloc_.Allocate();
    if (!page) {
      return nullptr;
    }
    page->id_ = page_id.id_;
    page->score_ = score;
    page_map_.Emplace(page_id, page);
    // Copy page from gcache (if exists)
    MemTask mem_task;
    mem_task.op_ = GcacheOp::kFault;
    mem_task.tag_id_ = bkt_.id_;
    mem_task.region_ = page_id.ChangePageSize(ctx_.gcache_page_size_);
    bool ret = gcache_->Read(mem_task, page->buf_ + page_id.off_);
    if (ret) {
      return page;
    }
    // Copy page into gcache
    gcache_->SubmitMemTask(mem_task);
    do {
      // Wait for page to be created
      ret = gcache_->Read(mem_task, page->buf_ + page_id.off_);
    } while (!ret);
    return page;
  }

  /** Flush page to gcache */
  HSHM_INLINE_GPU_FUN
  void FlushPage(const PageRegion &region) {
    Page *page = page_map_.Find(region);
    if (!page) {
      return;
    }
    MemTask mem_task;
    mem_task.op_ = GcacheOp::kFlush;
    mem_task.tag_id_ = bkt_.id_;
    mem_task.region_ = region.ChangePageSize(ctx_.gcache_page_size_);
    mem_task.score_ = page->score_;
    gcache_->Write(mem_task, page->buf_ + region.off_);
    SubmitMemTask(mem_task);
  }

  /** Evict page from tcache */
  HSHM_INLINE_GPU_FUN
  void InvalidatePage(const PageRegion &region) {
    Page *page = page_map_.Remove(region);
    if (page) {
      page_alloc_.Free(page);
    }
  }
};

template <typename T>
class VectorSet {
 public:
  hermes::Bucket bkt_;
  size_t size_ = 0;
  Vector<T> gpus_[HSHM_MAX_GPUS];

 public:
  VectorSet(const std::string &url, const Context &ctx = Context()) {
    bkt_ = hermes::Bucket(url);
    // for (int gpu_id = 0; gpu_id < CHI_CLIENT->ngpu_; ++gpu_id) {
    //   gpus_[gpu_id] = Vector<T>(bkt_, ctx, gpu_id);
    // }
  }

  void resize(size_t size) {
    // for (int gpu_id = 0; gpu_id < CHI_CLIENT->ngpu_; ++gpu_id) {
    //   gpus_[gpu_id].size_ = size;
    // }
    // size_ = size;
  }

  size_t size() const { return size_; }

  void Destroy() {
    // bkt_.Destroy();
  }

  Vector<T> &Get(int gpu_id) { return gpus_[gpu_id]; }
};

}  // namespace eternia

#endif  // ETERNIA_VECTOR_H