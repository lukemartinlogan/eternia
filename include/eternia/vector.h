#include <hermes/hermes.h>

#include "eternia/eternia_core_client.h"
#include "hermes_shm/constants/macros.h"
#include "libgpu.h"

namespace eternia {

#define MAX_TCACHE_SLOTS 32

struct Page {
  char *buf_;
  size_t size_;
  size_t id_;
  Page *next_;
};

template <size_t SIZE>
struct StaticPage {
  char buf_[SIZE];
  size_t size_ = SIZE;
  size_t id_;
  StaticPage *next_;
};

struct PageId {
  size_t id_;
  size_t off_;

  HSHM_INLINE_CROSS_FUN
  PageId(size_t off, size_t page_size) {
    id_ = off / page_size;
    off_ = off % page_size;
  }

  HSHM_INLINE_CROSS_FUN
  size_t Hash() const {
    size_t hash = 5381;
    hash = ((hash << 5) + hash) + id_; /* hash * 33 + c */
    return hash;
  }
};

struct Context {
  size_t gcache_page_size_ = MEGABYTES(1);
};

struct PageMap {
  Page *bkts_[MAX_TCACHE_SLOTS];

  PageMap() { memset(bkts_, 0, sizeof(bkts_)); }

  HSHM_INLINE_CROSS_FUN
  void Emplace(const PageId &page_id, Page *page) {
    size_t bkt_id = page_id.Hash() % MAX_TCACHE_SLOTS;
    Page *bkt = bkts_[bkt_id];
    page->next_ = bkt;
    bkts_[page_id.Hash() % MAX_TCACHE_SLOTS] = page;
  }

  HSHM_INLINE_CROSS_FUN
  Page *Find(const PageId &page_id) {
    size_t bkt_id = page_id.Hash() % MAX_TCACHE_SLOTS;
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
    size_t bkt_id = page_id.Hash() % MAX_TCACHE_SLOTS;
    Page *bkt = bkts_[bkt_id];
    if (bkt == nullptr) {
      return nullptr;
    }

    if (bkt->id_ == page_id.id_) {
      bkts_[page_id.Hash() % MAX_TCACHE_SLOTS] = bkt->next_;
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

template <typename T, size_t PAGE_SIZE = 4096,
          size_t PAGE_SLOTS = MAX_TCACHE_SLOTS>
class Vector {
 public:
  hermes::Bucket bkt_;
  Context ctx_;
  CHI_DATA_GPU_ALLOC_T *gpu_alloc_;
  PageAllocator<StaticPage<PAGE_SIZE>, PAGE_SLOTS> page_alloc_;
  PageMap page_map_;
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
    PageId page_id(off * sizeof(T), PAGE_SIZE);
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
    PageId page_id(off * sizeof(T), PAGE_SIZE);
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

template <typename T, size_t PAGE_SIZE = 4096,
          size_t PAGE_SLOTS = MAX_TCACHE_SLOTS>
class VectorSet {
 public:
  hermes::Bucket bkt_;
  Vector<T, PAGE_SIZE, PAGE_SLOTS> gpus_[MAX_GPU];

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