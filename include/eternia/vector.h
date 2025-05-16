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

#define TCACHE_TEMPLATE_PARAMS                            \
  size_t TCACHE_PAGE_SIZE = DEFAULT_TCACHE_PAGE_SIZE,     \
         size_t TCACHE_PAGE_SLOTS = DEFAULT_TCACHE_SLOTS, \
         size_t DEPTH = TCACHE_PAGE_SLOTS

#define TCACHE_TEMPLATE template <TCACHE_TEMPLATE_PARAMS>

#define TCACHE_TEMPLATE_PASS_ARGS TCACHE_PAGE_SIZE, TCACHE_PAGE_SLOTS

template <TCACHE_TEMPLATE_PARAMS>
struct StaticPage {
  char buf_[TCACHE_PAGE_SIZE];
  size_t id_;
  float score_;
  StaticPage *next_;
};

template <TCACHE_TEMPLATE_PARAMS>
struct StaticPageAllocator {
  typedef StaticPage<TCACHE_TEMPLATE_PASS_ARGS> Page;
  size_t head_ = 0, tail_ = 0;
  size_t PAGE_SIZE;
  Page pages_[DEPTH];
  Page *alloc_[DEPTH];

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  StaticPageAllocator() {
    memset(pages_, 0, sizeof(pages_));
    memset(alloc_, 0, sizeof(alloc_));
  }

  /** Allocate a page */
  HSHM_INLINE_CROSS_FUN Page *Allocate() {
    if (tail_ - head_ >= DEPTH) {
      return nullptr;
    } else if (tail_ < DEPTH) {
      size_t tail = tail_++;
      return (Page *)(pages_ + (tail * PAGE_SIZE));
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

template <TCACHE_TEMPLATE_PARAMS>
struct StaticPageMap {
  typedef StaticPage<TCACHE_TEMPLATE_PASS_ARGS> Page;
  Page *bkts_[DEPTH];

  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  StaticPageMap() { memset(bkts_, 0, sizeof(bkts_)); }

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

struct VectorCtx {
  hermes::Bucket bkt_;
  hipc::FullPtr<GpuCache> gcache_;
  size_t vec_size_ = 0;
  size_t tcache_page_size_ = DEFAULT_TCACHE_PAGE_SIZE;
  size_t tcache_page_slots_ = DEFAULT_TCACHE_SLOTS;
  size_t gcache_page_size_ = MEGABYTES(1);
};

template <typename T, TCACHE_TEMPLATE_PARAMS>
class Vector : public VectorCtx {
 public:
  typedef T Type;
  typedef StaticPage<TCACHE_TEMPLATE_PASS_ARGS> Page;
  typedef StaticPageAllocator<TCACHE_TEMPLATE_PASS_ARGS> PageAllocator;
  typedef StaticPageMap<TCACHE_TEMPLATE_PASS_ARGS> PageMap;

 public:
  PageAllocator *page_alloc_;
  PageMap *page_map_;
  Page *last_page_ = nullptr;
  size_t page_bnds_ = 0;  // A counter for number of page boundaries crossed

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Vector() = default;

  /** Emplace constructor */
  HSHM_INLINE_CROSS_FUN
  Vector(const VectorCtx &ctx) : VectorCtx(ctx) {
    page_alloc_ = (PageAllocator *)malloc(sizeof(PageAllocator));
    page_map_ = (PageMap *)malloc(sizeof(PageMap));
    gcache_ = hipc::FullPtr<GpuCache>(gcache_.shm_);
  }

  /** Get page allocator */
  HSHM_INLINE_CROSS_FUN
  PageAllocator &GetPageAlloc() { return *page_alloc_; }

  /** Get page map */
  HSHM_INLINE_CROSS_FUN
  PageMap &GetPageMap() { return *page_map_; }

  /** Get size */
  HSHM_INLINE_CROSS_FUN
  size_t size() const { return vec_size_; }

  /** Locally rescore a tcache page */
  HSHM_INLINE_GPU_FUN
  void SuggestScore(const PageRegion &region, float score) {
    Page *page = GetPageMap().Find(region);
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
    Page *page = GetPageMap().Find(region);
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
    mem_task.region_ = region.ChangePageSize(gcache_page_size_);
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
    PageRegion page_id(off * sizeof(T), tcache_page_size_);
    if (!FindPageInTcache(off, last_page_, page_id)) {
      return false;
    }
    val = GetValFromPage(page_id);
    return true;
  }

  /** Fault page from gcache into tcache */
  HSHM_INLINE_GPU_FUN
  Page *FaultPage(const PageRegion &page_id, float score) {
    // Make sure page is not already in tcache
    Page *page = GetPageMap().Find(page_id);
    if (page) {
      return page;
    }
    // Create page in tcache
    page = GetPageAlloc().Allocate();
    if (!page) {
      return nullptr;
    }
    page->id_ = page_id.id_;
    page->score_ = score;
    GetPageMap().Emplace(page_id, page);
    // Copy page from gcache (if exists)
    MemTask mem_task;
    mem_task.op_ = GcacheOp::kFault;
    mem_task.tag_id_ = bkt_.id_;
    mem_task.region_ = page_id.ChangePageSize(gcache_page_size_);
    bool ret = gcache_->Read(mem_task, page->buf_ + page_id.off_);
    if (ret) {
      return page;
    }
    // Copy page into gcache
    gcache_->SubmitMemTask(mem_task);
    do {
      // Wait for page to be created
      ret = gcache_->Read(mem_task, page->buf_ + page_id.off_);
      HSHM_THREAD_MODEL->Yield();
    } while (!ret);
    return page;
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
    page = GetPageMap().Find(page_id);
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
    PageRegion page_id(off, tcache_page_size_);
    Page *page = GetPageAlloc().Allocate();
    if (!page) {
      return nullptr;
    }
    GetPageMap().Emplace(page_id, page);
    return page;
  }

  /** Free page from tcache */
  HSHM_INLINE_GPU_FUN
  void FreePage(size_t off) {
    PageRegion page_id(off, tcache_page_size_);
    Page *page = GetPageMap().Remove(page_id);
    if (page == last_page_) {
      last_page_ = nullptr;
    }
  }

  /** Flush page to gcache */
  HSHM_INLINE_GPU_FUN
  void FlushPage(const PageRegion &region) {
    Page *page = GetPageMap().Find(region);
    if (!page) {
      return;
    }
    MemTask mem_task;
    mem_task.op_ = GcacheOp::kFlush;
    mem_task.tag_id_ = bkt_.id_;
    mem_task.region_ = region.ChangePageSize(gcache_page_size_);
    mem_task.score_ = page->score_;
    gcache_->Write(mem_task, page->buf_ + region.off_);
    SubmitMemTask(mem_task);
  }

  /** Evict page from tcache */
  HSHM_INLINE_GPU_FUN
  void InvalidatePage(const PageRegion &region) {
    Page *page = GetPageMap().Remove(region);
    if (page) {
      GetPageAlloc().Free(page);
    }
  }
};

template <typename T>
class VectorSet {
 public:
  hermes::Bucket bkt_;
  size_t size_ = 0;
  VectorCtx gpus_[HSHM_MAX_GPUS];

 public:
  VectorSet(const std::string &url, const VectorCtx &ctx = VectorCtx()) {
    bkt_ = hermes::Bucket(url);
    for (int gpu_id = 0; gpu_id < CHI_CLIENT->ngpu_; ++gpu_id) {
      VectorCtx &gpu_ctx = gpus_[gpu_id];
      gpu_ctx = ctx;
      gpu_ctx.bkt_ = bkt_;
    }
  }

  void resize(size_t size) {
    for (int gpu_id = 0; gpu_id < CHI_CLIENT->ngpu_; ++gpu_id) {
      gpus_[gpu_id].vec_size_ = size;
    }
    size_ = size;
  }

  size_t size() const { return size_; }

  void Destroy() { bkt_.Destroy(); }

  VectorCtx &Get(int gpu_id) { return gpus_[gpu_id]; }
};

}  // namespace eternia

#endif  // ETERNIA_VECTOR_H