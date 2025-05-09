#include <hermes/hermes.h>

#include "eternia/eternia_core_client.h"
#include "hermes_shm/constants/macros.h"
#include "libgpu.h"

namespace eternia {

#define DEFAULT_TCACHE_PAGE_SIZE 4096
#define DEFAULT_TCACHE_SLOTS 32

template <size_t SIZE>
struct StaticPage {
  char buf_[SIZE];
  size_t size_ = SIZE;
  size_t id_;
  StaticPage *next_;
};

struct Context {
  size_t gcache_page_size_ = MEGABYTES(1);
};

template <typename Page, size_t DEPTH>
struct PageAllocator {
  size_t head_ = 0, tail_ = 0;
  Page pages_[DEPTH];
  Page *alloc_[DEPTH];
  PageAllocator() { memset(pages_, 0, sizeof(pages_)); }

  HSHM_INLINE_CROSS_FUN
  Page *Allocate() {
    if (tail_ - head_ >= DEPTH) {
      return nullptr;
    } else if (tail_ < DEPTH) {
      size_t tail = tail_++;
      return pages_[tail];
    } else {
      size_t tail = (tail_++) % DEPTH;
      return alloc_[tail];
    }
  }

  HSHM_INLINE_CROSS_FUN
  void Free(Page *page) {
    size_t head = (head_++) % DEPTH;
    alloc_[head] = page;
  }
};

template <typename Page, size_t DEPTH>
struct PageMap {
  Page *bkts_[DEPTH];

  PageMap() { memset(bkts_, 0, sizeof(bkts_)); }

  HSHM_INLINE_CROSS_FUN
  void Emplace(const PageId &page_id, Page *page) {
    size_t bkt_id = page_id.Hash() % DEPTH;
    Page *bkt = bkts_[bkt_id];
    page->next_ = bkt;
    bkts_[page_id.Hash() % DEPTH] = page;
  }

  HSHM_INLINE_CROSS_FUN
  Page *Find(const PageId &page_id) {
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

  HSHM_INLINE_CROSS_FUN
  Page *Remove(const PageId &page_id) {
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

template <typename T, size_t TCACHE_PAGE_SIZE = DEFAULT_TCACHE_PAGE_SIZE,
          size_t TCACHE_PAGE_SLOTS = DEFAULT_TCACHE_SLOTS>
class Vector {
 public:
  typedef StaticPage<TCACHE_PAGE_SIZE> Page;

 public:
  hermes::Bucket bkt_;
  Context ctx_;
  CHI_DATA_GPU_ALLOC_T *gpu_alloc_;
  PageAllocator<Page, TCACHE_PAGE_SLOTS> page_alloc_;
  PageMap<Page, TCACHE_PAGE_SLOTS> page_map_;
  Page *last_page_;
  hipc::FullPtr<GpuCache> gcache_;

 public:
  Vector() = default;
  Vector(hermes::Bucket bkt, const Context &ctx, int gpu_id) {
    bkt_ = bkt;
    ctx_ = ctx;
    gpu_alloc_ = CHI_CLIENT->GetGpuDataAlloc(gpu_id);
    gcache_ = ETERNIA_CLIENT->gcache_[gpu_id];
    last_page_ = nullptr;
  }

  /**  */
  void Destroy() {}

  /** Find value at offset off in tcache.
   * @param off Offset is in units of T.
   * @param val Pointer to value at offset off.
   */
  HSHM_INLINE_GPU_FUN
  bool FindValInTcache(size_t off, T *&val) {
    PageId page_id(off * sizeof(T), TCACHE_PAGE_SIZE);
    last_page_ = FindPageInTcache(off, page_id);
    val = GetValFromPage(page_id);
    return false;
  }

  /** Find page containing offset off in tcache.
   * @param off Offset is in units of T.
   * @param page Pointer to page containing offset off.
   * @return True if the page is found in tcache, false otherwise.
   */
  HSHM_INLINE_GPU_FUN
  bool FindPageInTcache(size_t off, Page *&page) {
    PageId page_id(off * sizeof(T), TCACHE_PAGE_SIZE);
    return FindPageInTcache(off, page, page_id);
  }

  /** Find page containing offset off in tcache, given PageId
   * @param off Offset is in units of T.
   * @param page Pointer to page containing offset off.
   * @param page_id PageId of the page to find.
   * @return True if the page is found in tcache, false otherwise.
   */
  HSHM_INLINE_GPU_FUN
  bool FindPageInTcache(size_t off, Page *&page, const PageId &page_id) {
    // Check the last pointer
    if (last_page_ && last_page_->id_ == page_id.id_) {
      return true;
    }
    // Check the hash pointer
    page = page_map_.Find(page_id);
    return page != nullptr;
  }

 private:
  /** Gets the value in page using PageId */
  T *GetValFromPage(const PageId &page_id) {
    if (last_page_) {
      return (T *)(last_page_->buf_)[page_id.off_];
    }
    return nullptr;
  }

 public:
  HSHM_INLINE_GPU_FUN
  Page *AllocatePage(size_t off) {
    PageId page_id(off, ctx_.tcache_page_size_);
    Page *page = page_alloc_.Allocate();
    if (!page) {
      return nullptr;
    }
    page_map_.Emplace(page_id, page);
    return page;
  }

  HSHM_INLINE_GPU_FUN
  void EvictPage(size_t off) {
    PageId page_id(off, ctx_.tcache_page_size_);
    Page *page = page_map_.Remove(page_id);
    if (page == last_page_) {
      last_page_ = nullptr;
    }
  }

  HSHM_INLINE_GPU_FUN
  void PrefetchPage(size_t off) {}
};

template <typename T, size_t TCACHE_PAGE_SIZE = DEFAULT_TCACHE_PAGE_SIZE,
          size_t TCACHE_PAGE_SLOTS = DEFAULT_TCACHE_SLOTS>
class VectorSet {
 public:
  hermes::Bucket bkt_;
  Vector<T, TCACHE_PAGE_SIZE, TCACHE_PAGE_SLOTS> gpus_[MAX_GPU];

 public:
  VectorSet(const std::string &url, const Context &ctx = Context()) {
    bkt_ = hermes::Bucket(url);
    for (int gpu_id = 0; gpu_id < CHI_CLIENT->ngpu_; ++gpu_id) {
      gpus_[gpu_id] = Vector<T>(bkt_, ctx, gpu_id);
    }
  }

  void Destroy() { bkt_.Destroy(); }

  Vector<T> &Get(int gpu_id) { return gpus_[gpu_id]; }
};

}  // namespace eternia