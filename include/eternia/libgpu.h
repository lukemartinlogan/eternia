#ifndef ETERNIA_LIBGPU_H
#define ETERNIA_LIBGPU_H

#include <hermes/hermes.h>
#include <hermes_shm/constants/macros.h>

#include "constants.h"

#define GCACHE_MIN_SCORE 0.8

namespace eternia {

enum class GcacheOp { kRescore, kFault, kFlush, kEvict };

struct PageRegion {
  size_t id_;
  size_t off_;
  size_t size_;
  size_t page_size_;
  bool modified_;

  HSHM_INLINE_CROSS_FUN
  PageRegion() {}

  HSHM_INLINE_CROSS_FUN
  PageRegion(size_t off, size_t page_size) {
    id_ = off / page_size;
    off_ = off % page_size;
    size_ = page_size - off_;
    page_size_ = page_size;
  }

  HSHM_INLINE_CROSS_FUN
  PageRegion(size_t off, size_t size, size_t page_size, bool modified = false) {
    id_ = off / page_size;
    off_ = off % page_size;
    size_ = size;
    page_size_ = page_size;
    modified_ = modified;
  }

  HSHM_INLINE_CROSS_FUN
  size_t ToIndex() const { return id_ * page_size_ + off_; }

  HSHM_INLINE_CROSS_FUN
  PageRegion ChangePageSize(size_t new_size) const {
    return PageRegion(ToIndex(), size_, new_size, modified_);
  }

  HSHM_INLINE_CROSS_FUN
  size_t Hash() const {
    size_t hash = 5381;
    hash = ((hash << 5) + hash) + id_; /* hash * 33 + c */
    return hash;
  }
};

struct MemTask {
  hermes::TagId tag_id_;
  GcacheOp op_;
  PageRegion region_;
  float score_;

  HSHM_INLINE_CROSS_FUN size_t Hash() const { return Hash1(); }

  HSHM_INLINE_CROSS_FUN
  size_t Hash1() const {
    // Jenkins one at a time hash
    size_t hash = 0;
    hash += tag_id_.unique_;
    hash += (hash << 10);
    hash ^= (hash >> 6);
    hash += region_.id_;
    hash += (hash << 10);
    hash ^= (hash >> 6);
    hash += tag_id_.node_id_;
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
  }

  HSHM_INLINE_CROSS_FUN
  size_t Hash2() const {
    size_t result = 17;
    result = result * 31 + tag_id_.unique_;
    result = result * 31 + tag_id_.node_id_;
    result = result * 31 + region_.id_;
    return result;
  }

  HSHM_INLINE_CROSS_FUN
  size_t Hash3() const {
    size_t hash = 14695981039346656037UL;
    hash = (hash ^ tag_id_.unique_) * 1099511628211UL;
    hash = (hash ^ tag_id_.node_id_) * 1099511628211UL;
    hash = (hash ^ region_.id_) * 1099511628211UL;
    return hash;
  }
};

struct Metadata {
  hermes::TagId tag_id_;
  size_t region_;
  chi::Block block_;
  hipc::Pointer data_ = hipc::Pointer::GetNull();
  int dev_id_;
};

class GpuCache {
 public:
  chi::ipc::mpsc_queue<MemTask> cpu_queue_;
  chi::data::ipc::vector<chi::data::ipc::mpsc_queue<MemTask>> gpu_queues_;
  chi::data::ipc::vector<Metadata> md_cache_;
  hipc::RwLock md_lock_;
  hermes::Client mdm_;

 public:
  HSHM_GPU_FUN
  GpuCache(int count, int depth, hermes::Client mdm)
      : gpu_queues_(CHI_CLIENT->data_alloc_, count, depth),
        cpu_queue_(CHI_CLIENT->main_alloc_, depth),
        md_cache_(CHI_CLIENT->data_alloc_, count * depth),
        mdm_(mdm) {}

 public:
  /** Write to gcache */
  HSHM_GPU_FUN
  bool Write(const MemTask &mem_task, char *buf) {
    const PageRegion &region = mem_task.region_;
    hipc::ScopedRwReadLock lock(md_lock_, 0);
    Metadata *md;
    if (Find(mem_task, md)) {
      // Write to the page
      hipc::FullPtr<char> page(md->data_);
      size_t off = region.off_;
      memcpy(page.ptr_ + off, buf, region.size_);
      return true;
    }
    return false;
  }

  /** Read from gcache */
  HSHM_GPU_FUN
  bool Read(const MemTask &mem_task, char *buf) {
    Metadata *md;
    const PageRegion &region = mem_task.region_;
    hipc::ScopedRwReadLock lock(md_lock_, 0);
    if (Find(mem_task, md)) {
      // Read from the page
      hipc::FullPtr<char> page(md->data_);
      size_t off = region.off_;
      memcpy(buf, page.ptr_ + off, region.size_);
      return true;
    }
    return false;
  }

  /** Submit a memory task to the GCache */
  HSHM_INLINE_CROSS_FUN
  void SubmitMemTask(const MemTask &mem_task) {
#ifdef HSHM_IS_HOST
    cpu_queue_.push(mem_task);
#else
    size_t hash = mem_task.Hash();
    size_t id = hash % gpu_queues_.size();
    gpu_queues_[id].push(mem_task);
#endif
  }

  /** Execute memory task */
  HSHM_GPU_FUN void ProcessMemTask(MemTask *mem_task) {
    switch (mem_task->op_) {
      case GcacheOp::kRescore:
        // Rescore the page
        Rescore(*mem_task);
        break;
      case GcacheOp::kFault:
        Fault(*mem_task);
        break;
      case GcacheOp::kFlush:
        Flush(*mem_task);
        break;
      case GcacheOp::kEvict:
        Evict(*mem_task);
        break;
    }
  }

 private:
  /** Rescore pages */
  HSHM_GPU_FUN
  void Rescore(const MemTask &mem_task) {
    if (mem_task.score_ >= GCACHE_MIN_SCORE) {
      Fault(mem_task);
    } else if (mem_task.score_ > 0.0) {
      Evict(mem_task);
      FaultMd(mem_task);
    } else {
      Evict(mem_task);
    }
  }

  /** Fault md + data into gcache */
  HSHM_GPU_FUN
  bool Fault(const MemTask &mem_task) {
    Metadata *md;
    // Create metadata for page
    if (!Find(mem_task, md)) {
      GetOrCreateMetadata(mem_task, md);
    }
    // Check if the data pointer is valid
    if (!md->data_.IsNull()) {
      return true;
    }
    // Fault the page
    hipc::FullPtr<char> page(md->data_);
    const PageRegion &region = mem_task.region_;
    hipc::ScopedTlsAllocator<CHI_MAIN_ALLOC_T> tls(CHI_CLIENT->main_alloc_);
    hermes::Context ctx;
    ctx.mctx_ = tls.alloc_.ctx_;
    hermes::Bucket bkt(mem_task.tag_id_, mdm_, ctx);
    hermes::Blob blob(md->data_ + region.off_, region.size_, false);
    chi::string blob_name(CHI_CLIENT->main_alloc_, sizeof(size_t));
    memcpy(blob_name.data(), &region.id_, sizeof(size_t));
    bkt.PartialGet(blob_name, blob, region.off_);
    return true;
  }

  /** Fault only metadata into gcache */
  HSHM_GPU_FUN
  bool FaultMd(const MemTask &mem_task) {
    Metadata *md;
    // Create metadata for page
    if (!Find(mem_task, md)) {
      GetOrCreateMetadata(mem_task, md);
    }
    // Fault the page
    hipc::FullPtr<char> page(md->data_);
    size_t off = mem_task.region_.off_;
    hermes::Bucket bkt(mem_task.tag_id_, mdm_);
    // bkt.PartialGet()
    return true;
  }

  /** Get or create a metadata entry */
  HSHM_GPU_FUN
  bool GetOrCreateMetadata(const MemTask &mem_task, Metadata *&md) {
    hipc::ScopedRwWriteLock lock(md_lock_, 0);
    // Check if the page is already in gcache
    if (Find(mem_task, md)) {
      return true;
    }
    // Create metadata for page
    size_t hash = mem_task.Hash();
    size_t id = hash % gpu_queues_.size();
    md = &md_cache_[id];
    return false;
  }

  /** Evict the page from gcache */
  HSHM_INLINE_GPU_FUN void Evict(const MemTask &mem_task) {
    Flush(mem_task);
    Invalidate(mem_task);
  }

  /** Delete entry from gcache */
  HSHM_GPU_FUN
  bool Invalidate(const MemTask &mem_task) {
    Metadata *md;
    // Don't do partial invalidations
    if (mem_task.region_.page_size_ != mem_task.region_.size_) {
      return false;
    }
    // Find metadata to evict
    hipc::ScopedRwWriteLock lock(md_lock_, 0);
    if (Find(mem_task, md)) {
      return true;
    }
    return false;
  }

  /** Flush entry to scache */
  HSHM_GPU_FUN
  bool Flush(const MemTask &mem_task) {
    Metadata *md;
    hipc::ScopedRwReadLock lock(md_lock_, 0);
    if (!Find(mem_task, md)) {
      return false;
    }
    // Flush the page
    hipc::FullPtr<char> page(md->data_);
    const PageRegion &region = mem_task.region_;
    hipc::ScopedTlsAllocator<CHI_MAIN_ALLOC_T> tls(CHI_CLIENT->main_alloc_);
    hermes::Context ctx;
    ctx.mctx_ = tls.alloc_.ctx_;
    hermes::Bucket bkt(mem_task.tag_id_, mdm_, ctx);
    hermes::Blob blob(md->data_ + region.off_, region.size_, false);
    chi::string blob_name(CHI_CLIENT->main_alloc_, sizeof(size_t));
    memcpy(blob_name.data(), &region.id_, sizeof(size_t));
    bkt.PartialPut(blob_name, blob, region.off_);
    return false;
  }

  /** Find the metadata associated with MemTask page */
  HSHM_INLINE_GPU_FUN
  bool Find(const MemTask &mem_task, Metadata *&md) {
    return Find(mem_task, md, mem_task.Hash());
  }

  /** Find the metadata associated with MemTask page */
  HSHM_GPU_FUN
  bool Find(const MemTask &mem_task, Metadata *&md, size_t mem_task_hash) {
    size_t id = mem_task_hash % md_cache_.size();
    md = &md_cache_[id];
    if (md->tag_id_ == mem_task.tag_id_ &&
        md->region_ == mem_task.region_.id_) {
      return true;
    }
    return false;
  }
};

}  // namespace eternia

#endif