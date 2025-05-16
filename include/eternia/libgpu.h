#ifndef ETERNIA_LIBGPU_H
#define ETERNIA_LIBGPU_H

#include <hermes/hermes.h>
#include <hermes_shm/constants/macros.h>
#include <hermes_shm/util/gpu_api.h>

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

  HSHM_INLINE_CROSS_FUN
  size_t Hash() const {
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
  size_t operator()(const MemTask &obj) const { return obj.Hash(); }

  HSHM_INLINE_CROSS_FUN
  bool operator==(const MemTask &other) const {
    return tag_id_ == other.tag_id_ && op_ == other.op_ &&
           region_.id_ == other.region_.id_;
  }

  HSHM_INLINE_CROSS_FUN
  bool operator!=(const MemTask &other) const { return !(*this == other); }
};

struct Metadata {
  hermes::TagId tag_id_;
  size_t page_id_;
  chi::Block block_;
  hipc::Pointer data_ = hipc::Pointer::GetNull();
  int dev_id_;
};

struct MdBucket : public hipc::ShmContainer {
  typedef chi::data::ipc::slist<Metadata> MD_LIST_T;
  chi::data::ipc::slist<Metadata> md_list_;
  hipc::RwLock md_lock_;

  HSHM_INLINE_CROSS_FUN
  MdBucket(const hipc::CtxAllocator<CHI_DATA_ALLOC_T> &alloc)
      : md_list_(alloc) {}

  HSHM_INLINE_CROSS_FUN
  MdBucket(const hipc::CtxAllocator<CHI_DATA_ALLOC_T> &alloc,
           const MdBucket &other)
      : ShmContainer(), md_list_(alloc, other.md_list_), md_lock_() {}
};

class GpuCache {
 public:
  typedef chi::ipc::mpsc_queue<MemTask> CPU_QUEUE_T;
  typedef chi::data::ipc::mpsc_queue<MemTask> GPU_QUEUE_T;
  typedef chi::data::ipc::vector<GPU_QUEUE_T> GPU_QUEUE_MAP_T;
  typedef chi::data::ipc::vector<MdBucket> MD_CACHE_T;
  typedef MdBucket::MD_LIST_T::iterator_t MD_LIST_ITER_T;
  typedef chi::slist<MemTask> MEM_TASK_SET_T;
  typedef chi::unordered_map<MemTask, MEM_TASK_SET_T, MemTask> AGG_MAP_T;

 public:
  CPU_QUEUE_T cpu_queue_;
  GPU_QUEUE_MAP_T gpu_queues_;
  MD_CACHE_T md_cache_;
  hermes::Client mdm_;
  int nthreads_;  // Number of gcache workers

#ifdef HSHM_ENABLE_CUDA_OR_ROCM
 public:
  HSHM_GPU_FUN
  GpuCache(int count, int depth, hermes::Client mdm)
      : nthreads_(count),
        gpu_queues_(CHI_CLIENT->data_alloc_, count, depth),
        cpu_queue_(CHI_CLIENT->main_alloc_, depth),
        md_cache_(CHI_CLIENT->data_alloc_, count * depth),
        mdm_(mdm) {}

 public:
  /** Write to gcache */
  HSHM_GPU_FUN
  bool Write(const MemTask &mem_task, char *buf) {
    const PageRegion &region = mem_task.region_;
    // Find bucket + metadata
    MdBucket *md_bkt;
    if (!FindBucket(mem_task, md_bkt)) {
      return false;
    }
    Metadata *md;
    hipc::ScopedRwReadLock lock(md_bkt->md_lock_, 0);
    if (!FindMetadata(mem_task, md_bkt, md)) {
      return false;
    }
    // Write to the page
    hipc::FullPtr<char> page(md->data_);
    size_t off = region.off_;
    memcpy(page.ptr_ + off, buf, region.size_);
    return true;
  }

  /** Read from gcache */
  HSHM_GPU_FUN
  bool Read(const MemTask &mem_task, char *buf) {
    const PageRegion &region = mem_task.region_;
    // Find bucket + metadata
    MdBucket *md_bkt;
    if (!FindBucket(mem_task, md_bkt)) {
      return false;
    }
    Metadata *md;
    hipc::ScopedRwReadLock lock(md_bkt->md_lock_, 0);
    if (!FindMetadata(mem_task, md_bkt, md)) {
      return false;
    }
    // Read from the page
    hipc::FullPtr<char> page(md->data_);
    size_t off = region.off_;
    memcpy(buf, page.ptr_ + off, region.size_);
    return true;
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

  /** Aggregate memory task */
  HSHM_GPU_FUN void AggregateTask(AGG_MAP_T &agg_map, MemTask *mem_task) {
    agg_map[*mem_task].emplace_back(*mem_task);
  }

  /** Process memory tasks */
  HSHM_GPU_FUN void ProcessMemTasks(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &ctx_alloc,
      AGG_MAP_T &agg_map) {
    for (auto it = agg_map.begin(); it != agg_map.end(); ++it) {
      MEM_TASK_SET_T &to_agg = (*it).GetVal();
      // Merge aggregated tasks
      printf("Beginning to merge tasks\n");
      chi::vector<MemTask> merged = MergeMemTasks(ctx_alloc, to_agg);
      printf("Merged tasks\n");
      // Process the merged tasks
      for (const MemTask &mem_task : merged) {
        ProcessMemTask(ctx_alloc, mem_task);
      }
    }
  }

 private:
  /** Merge a group of memory tasks */
  HSHM_GPU_FUN
  chi::vector<MemTask> MergeMemTasks(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &ctx_alloc,
      MEM_TASK_SET_T &to_agg) {
    // Copy memtasks to vector
    chi::vector<MemTask> to_agg_sorted(ctx_alloc, to_agg.size());
    size_t i = 0;
    for (MemTask &mem_task : to_agg) {
      to_agg_sorted[i] = mem_task;
    }
    // Sort the memtasks by offset
    hshm::insertion_sort(to_agg_sorted.begin(), to_agg_sorted.end(),
                         [](const MemTask &a, const MemTask &b) {
                           return a.region_.off_ < b.region_.off_;
                         });
    // Aggregate memtasks
    chi::vector<MemTask> agg(ctx_alloc);
    agg.reserve(to_agg_sorted.size());
    agg.emplace_back(to_agg_sorted[0]);
    for (size_t i = 1; i < to_agg_sorted.size(); ++i) {
      MemTask &cur = agg.back();
      MemTask &next = to_agg_sorted[i];
      size_t cur_region_end = cur.region_.off_ + cur.region_.size_;
      size_t next_region_end = next.region_.off_ + next.region_.size_;
      if (cur_region_end >= next.region_.off_) {
        cur.region_.size_ = next_region_end - cur.region_.off_;
      } else {
        agg.emplace_back(cur);
      }
    }
    return agg;
  }

  /** Process a memory task */
  HSHM_GPU_FUN
  void ProcessMemTask(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &ctx_alloc,
                      const MemTask &mem_task) {
    switch (mem_task.op_) {
      case GcacheOp::kRescore:
        // Rescore the page
        Rescore(ctx_alloc, mem_task);
        break;
      case GcacheOp::kFault:
        Fault(ctx_alloc, mem_task);
        break;
      case GcacheOp::kFlush:
        Flush(ctx_alloc, mem_task);
        break;
      case GcacheOp::kEvict:
        Evict(ctx_alloc, mem_task);
        break;
    }
  }

  /** Rescore pages */
  HSHM_GPU_FUN
  void Rescore(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &ctx_alloc,
               const MemTask &mem_task) {
    if (mem_task.score_ >= GCACHE_MIN_SCORE) {
      Fault(ctx_alloc, mem_task);
    } else {
      Evict(ctx_alloc, mem_task);
    }
  }

  /** Make blob name */
  HSHM_GPU_FUN
  chi::string MakeBlobName(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &ctx_alloc,
      const PageRegion &region) {
    chi::string blob_name(ctx_alloc, sizeof(size_t));
    memcpy(blob_name.data(), &region.id_, sizeof(size_t));
    return blob_name;
  }

  /** Fault md + data into gcache */
  HSHM_GPU_FUN
  bool Fault(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &ctx_alloc,
             const MemTask &mem_task) {
    // Find or create bucket + metadata
    MdBucket *md_bkt;
    if (!FindBucket(mem_task, md_bkt)) {
      return false;
    }
    Metadata *md;
    hipc::ScopedRwReadLock lock(md_bkt->md_lock_, 0);
    if (!FindMetadata(mem_task, md_bkt, md)) {
      md_bkt->md_lock_.ReadUnlock();
      CreateMetadata(ctx_alloc, mem_task, md);
      md_bkt->md_lock_.ReadLock(0);
    }
    // Check if the data pointer is valid
    if (!md->data_.IsNull()) {
      return true;
    }
    // Fault the page
    hipc::FullPtr<char> page(md->data_);
    const PageRegion &region = mem_task.region_;
    auto *main_alloc = CHI_CLIENT->main_alloc_;
    hermes::Context ctx;
    ctx.mctx_ = ctx_alloc.ctx_;
    hermes::Bucket bkt(mem_task.tag_id_, mdm_, ctx);
    hermes::Blob blob(md->data_ + region.off_, region.size_, false);
    chi::string blob_name = MakeBlobName(ctx_alloc, region);
    bkt.PartialGet(blob_name, blob, region.off_);
    return true;
  }

  /** Get or create a metadata entry */
  HSHM_GPU_FUN
  bool CreateMetadata(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &ctx_alloc,
                      const MemTask &mem_task, Metadata *&md) {
    // Find bucket + metadata
    MdBucket *md_bkt;
    if (!FindBucket(mem_task, md_bkt)) {
      return false;
    }
    // Create metadata for page
    hipc::ScopedRwWriteLock lock(md_bkt->md_lock_, 0);
    md_bkt->md_list_.emplace_back();
    md = &md_bkt->md_list_.back();
    md->tag_id_ = mem_task.tag_id_;
    md->page_id_ = mem_task.region_.id_;
    md->data_ =
        CHI_CLIENT->AllocateBuffer(HSHM_MCTX, mem_task.region_.page_size_).shm_;
    return true;
  }

  /** Evict the page from gcache */
  HSHM_INLINE_GPU_FUN void Evict(
      const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &ctx_alloc,
      const MemTask &mem_task) {
    Flush(ctx_alloc, mem_task);
    Invalidate(ctx_alloc, mem_task);
  }

  /** Delete entry from gcache */
  HSHM_GPU_FUN
  bool Invalidate(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &ctx_alloc,
                  const MemTask &mem_task) {
    Metadata *md;
    // Don't do partial invalidations
    if (mem_task.region_.page_size_ != mem_task.region_.size_) {
      return false;
    }
    // Find bucket
    MdBucket *md_bkt;
    if (!FindBucket(mem_task, md_bkt)) {
      return false;
    }
    // Find metadata to evict
    hipc::ScopedRwWriteLock lock(md_bkt->md_lock_, 0);
    auto it = FindMetadata(mem_task, md_bkt);
    if (it != md_bkt->md_list_.end()) {
      md_bkt->md_list_.erase(it);
      return true;
    }
    return false;
  }

  /** Flush entry to scache */
  HSHM_GPU_FUN
  bool Flush(const hipc::CtxAllocator<CHI_MAIN_ALLOC_T> &ctx_alloc,
             const MemTask &mem_task) {
    // Find bucket + metadata
    MdBucket *md_bkt;
    if (!FindBucket(mem_task, md_bkt)) {
      return false;
    }
    Metadata *md;
    hipc::ScopedRwReadLock lock(md_bkt->md_lock_, 0);
    if (!FindMetadata(mem_task, md_bkt, md)) {
      return false;
    }
    // Flush the page
    hipc::FullPtr<char> page(md->data_);
    const PageRegion &region = mem_task.region_;
    auto *main_alloc = CHI_CLIENT->main_alloc_;
    hermes::Context ctx;
    ctx.mctx_ = ctx_alloc.ctx_;
    hermes::Bucket bkt(mem_task.tag_id_, mdm_, ctx);
    hermes::Blob blob(md->data_ + region.off_, region.size_, false);
    chi::string blob_name = MakeBlobName(ctx_alloc, region);
    bkt.PartialPut(blob_name, blob, region.off_);
    return false;
  }

  /** Find the metadata associated with MemTask page */
  HSHM_INLINE_GPU_FUN
  bool FindBucket(const MemTask &mem_task, MdBucket *&bkt) {
    size_t mem_task_hash = mem_task.Hash();
    size_t id = mem_task_hash % md_cache_.size();
    bkt = &md_cache_[id];
    return true;
  }

  /** Find the metadata associated with MemTask page */
  HSHM_INLINE_GPU_FUN
  bool FindMetadata(const MemTask &mem_task, MdBucket *md_bkt,
                    Metadata *&md_ret) {
    for (Metadata &md : md_bkt->md_list_) {
      if (md.tag_id_ == mem_task.tag_id_ &&
          md.page_id_ == mem_task.region_.id_) {
        md_ret = &md;
        return true;
      }
    }
    md_ret = nullptr;
    return false;
  }

  /** Find the metadata associated with MemTask page */
  HSHM_INLINE_GPU_FUN
  MD_LIST_ITER_T FindMetadata(const MemTask &mem_task, MdBucket *md_bkt) {
    for (auto it = md_bkt->md_list_.begin(); it != md_bkt->md_list_.end();
         ++it) {
      Metadata &md = *it;
      if (md.tag_id_ == mem_task.tag_id_ &&
          md.page_id_ == mem_task.region_.id_) {
        return it;
      }
    }
    return md_bkt->md_list_.end();
  }
#endif
};

}  // namespace eternia

#endif