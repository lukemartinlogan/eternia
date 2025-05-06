#include <hermes/hermes.h>

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
  size_t tcache_size_ = KILOBYTES(64);
  size_t tcache_page_size_ = KILOBYTES(8);
  size_t gcache_page_size_ = MEGABYTES(1);
};

struct PageAllocator {
  size_t head_ = 0, tail_ = 0;
  hipc::FullPtr<char> tcache_;
  Context ctx_;
  Page pages_[MAX_TCACHE_SLOTS];
  Page *alloc_[MAX_TCACHE_SLOTS];

  PageAllocator() { memset(pages_, 0, sizeof(pages_)); }

  void init(hipc::FullPtr<char> tcache, Context ctx) {
    tcache_ = tcache;
    ctx_ = ctx;
  }

  HSHM_INLINE_CROSS_FUN
  Page *Allocate() {
    if (tail_ - head_ >= MAX_TCACHE_SLOTS) {
      return nullptr;
    } else if (tail_ < MAX_TCACHE_SLOTS) {
      size_t tail = tail_++;
      Page &page = pages_[tail];
      page.size_ = ctx_.tcache_page_size_;
      page.buf_ = tcache_.ptr_;
      tcache_ += ctx_.tcache_size_;
      return &page;
    } else {
      size_t tail = (tail_++) % MAX_TCACHE_SLOTS;
      return alloc_[tail];
    }
  }

  HSHM_INLINE_CROSS_FUN
  void Free(Page *page) {
    size_t head = (head_++) % MAX_TCACHE_SLOTS;
    alloc_[head] = page;
  }
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

template <typename T>
class Vector {
 public:
  hermes::Bucket bkt_;
  Context ctx_;
  CHI_DATA_GPU_ALLOC_T *gpu_alloc_;
  hipc::FullPtr<char> tcache_;
  PageAllocator page_alloc_;
  PageMap page_map_;
  Page *last_page_;

 public:
  Vector() = default;
  Vector(hermes::Bucket bkt, const Context &ctx, int gpu_id) {
    bkt_ = bkt;
    ctx_ = ctx;
    gpu_alloc_ = CHI_CLIENT->GetGpuDataAlloc(gpu_id);
    tcache_ = gpu_alloc_->AllocateLocalPtr<char>(HSHM_MCTX, ctx.tcache_size_);
    last_page_ = nullptr;
  }

  void Destroy() { gpu_alloc_->Free(HSHM_MCTX, tcache_); }

  HSHM_INLINE_GPU_FUN
  bool FindValInTcache(size_t off, T *&val) {
    PageId page_id(off, ctx_.tcache_page_size_);
    last_page_ = FindPageInTcache(off, page_id);
    val = GetValFromPage(page_id);
    return false;
  }

  HSHM_INLINE_GPU_FUN
  bool FindPageInTcache(size_t off, Page *&page) {
    PageId page_id(off, ctx_.tcache_page_size_);
    return FindPageInTcache(off, page, page_id);
  }

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
  T *GetValFromPage(const PageId &page_id) {
    if (last_page_) {
      return ((T *)last_page_->buf_)[page_id.off_];
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

template <typename T>
class VectorSet {
 public:
  hermes::Bucket bkt_;
  Vector<T> gpus_[MAX_GPU];

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