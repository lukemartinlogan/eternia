#include <hermes/hermes.h>

#include "hermes_shm/constants/macros.h"

namespace eternia {

enum class IoOp { kRead, kWrite, kAppend, kEvict };

struct MemTask {
  hermes::TagId tag_id_;
  IoOp op_;
  size_t off_;
  u32 size_;
  u32 page_size_;
};

template <typename T, size_t DEPTH>
struct PageAllocator {
  size_t head_ = 0, tail_ = 0;
  T pages_[DEPTH];
  T *alloc_[DEPTH];
  PageAllocator() { memset(pages_, 0, sizeof(pages_)); }

  HSHM_INLINE_CROSS_FUN
  T *Allocate() {
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
  void Free(T *page) {
    size_t head = (head_++) % DEPTH;
    alloc_[head] = page;
  }
};

struct ChunkId {
  hermes::TagId tag_id_;
  size_t page_id_;

  ChunkId(const MemTask &mem_task, hermes::TagId &tag_id) {
    page_id_ = mem_task.off_ / mem_task.page_size_;
    tag_id_ = tag_id;
  }

  HSHM_INLINE_CROSS_FUN
  size_t Hash1() {
    // Jenkins one at a time hash
    size_t hash = 0;
    hash += tag_id_.unique_;
    hash += (hash << 10);
    hash ^= (hash >> 6);
    hash += page_id_;
    hash += (hash << 10);
    hash ^= (hash >> 6);
    hash += tag_id_.node_id_;
    hash += (hash << 3);
    hash ^= (hash >> 11);
    hash += (hash << 15);
    return hash;
  }

  HSHM_INLINE_CROSS_FUN
  size_t Hash2() {
    size_t result = 17;
    result = result * 31 + tag_id_.unique_;
    result = result * 31 + tag_id_.node_id_;
    result = result * 31 + page_id_;
    return result;
  }

  HSHM_INLINE_CROSS_FUN
  size_t Hash3() {
    size_t hash = 14695981039346656037UL;
    hash = (hash ^ tag_id_.unique_) * 1099511628211UL;
    hash = (hash ^ tag_id_.node_id_) * 1099511628211UL;
    hash = (hash ^ page_id_) * 1099511628211UL;
    return hash;
  }
};

struct Metadata {
  chi::Block block_;
  hipc::Pointer data_;
  int dev_id_;
  hipc::atomic<bool> taken_;
};

class GpuCache {
 public:
  chi::data::ipc::vector<chi::data::ipc::mpsc_queue<MemTask>> gpu_queues_;
  chi::ipc::mpsc_queue<MemTask> cpu_queue_;
  chi::ipc::vector<Metadata> md_cache_;
  hermes::Client mdm_;

 public:
  GpuCache(int count, int depth, hermes::Client mdm)
      : gpu_queues_(count, depth),
        cpu_queue_(depth),
        md_cache_(count * depth),
        mdm_(mdm) {}

 public:
  template <bool IsTcache>
  HSHM_GPU_FUN void ProcessMemTask(MemTask *mem_task) {
    Metadata *md;
    if (Find(*mem_task, md)) {
    } else {
      // hermes::Bucket bkt(mem_task.tag_id_, mdm_);
    }
  }

  HSHM_GPU_FUN
  bool Find(const MemTask &mem_task, Metadata *&data) { return false; }

  template <typename T>
  void Emplace(const hermes::Bucket &bkt) {}

  template <typename T>
  void Evict() {}
};

}  // namespace eternia