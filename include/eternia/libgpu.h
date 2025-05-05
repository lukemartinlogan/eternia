#include <hermes/hermes.h>

#include "hermes_shm/constants/macros.h"

namespace eternia {

enum class IoOp { kRead, kWrite, kAppend };

struct MemTask {
  hermes::TagId tag_id_;
  IoOp op_;
  size_t off_;
  u32 size_;
  u32 page_size_;
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

struct GpuCacheAllocHdr {
  hermes::Client mdm_;
  hipc::delay_ar<chi::ipc::vector<Metadata>> md_cache_;
};

class GpuCache {
 public:
  hermes::Client mdm_;
  chi::ipc::vector<Metadata> *md_cache_;

 public:
  HSHM_GPU_FUN
  void shm_init(hermes::Client mdm) { mdm_ = mdm; }

  HSHM_GPU_FUN
  void shm_deserialize() {}

  HSHM_GPU_FUN void Io(const MemTask &mem_task) {
    Metadata *md;
    if (Find(mem_task, md)) {
    } else {
      hermes::Bucket bkt(mem_task.tag_id_, mdm_);
    }
  }

  HSHM_GPU_FUN
  bool Find(const MemTask &mem_task, Metadata *&data) {}

  template <typename T>
  void Emplace(const hermes::Bucket &bkt) {}

  template <typename T>
  void Evict() {}
};

}  // namespace eternia