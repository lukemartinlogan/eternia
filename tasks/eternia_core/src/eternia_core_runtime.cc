/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Distributed under BSD 3-Clause license.                                   *
 * Copyright by The HDF Group.                                               *
 * Copyright by the Illinois Institute of Technology.                        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of Hermes. The full Hermes copyright notice, including  *
 * terms governing use, modification, and redistribution, is contained in    *
 * the COPYING file, which can be found at the top directory. If you do not  *
 * have access to the file, you may request a copy from help@hdfgroup.org.   *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <chimaera/api/chimaera_runtime.h>
#include <chimaera/monitor/monitor.h>
#include <chimaera_admin/chimaera_admin_client.h>
#include <hermes/hermes.h>
#include <hermes_shm/util/gpu_api.h>

#include "eternia_core/eternia_core_client.h"
#include "eternia_core/eternia_core_tasks.h"

namespace eternia {

HSHM_GPU_KERNEL void InitEterniaRuntime(int qcount, int qdepth,
                                        FullPtr<GpuCache> *gcache,
                                        hermes::Client client) {
  hipc::ScopedTlsAllocator<CHI_DATA_ALLOC_T> tls(CHI_CLIENT->data_alloc_);
  *gcache = tls.alloc_->NewObjLocal<GpuCache>(tls.alloc_.ctx_, qcount, qdepth,
                                              client);
}

template <typename QueueT>
HSHM_GPU_FUN void PollEterniaQueue(hipc::FullPtr<GpuCache> gcache,
                                   QueueT &queue) {
  size_t count = queue.size();
  for (size_t i = 0; i < count; ++i) {
    MemTask task;
    if (queue.pop(task).IsNull()) {
      break;
    }
    gcache->ProcessMemTask(&task);
  }
}

HSHM_GPU_KERNEL void PollCpuQueue(hipc::Pointer gcache_p) {
  hipc::FullPtr<GpuCache> gcache(gcache_p);
  size_t tid = hshm::GpuApi::GetGlobalThreadId();
  if (tid == 0) {
    PollEterniaQueue(gcache, gcache->cpu_queue_);
  }
}

HSHM_GPU_KERNEL void PollGpuQueues(hipc::Pointer gcache_p) {
  hipc::FullPtr<GpuCache> gcache(gcache_p);
  size_t tid = hshm::GpuApi::GetGlobalThreadId();
  if (tid < gcache->gpu_queues_.size()) {
    PollEterniaQueue(gcache, gcache->gpu_queues_[tid]);
  }
}

class Server : public Module {
 public:
  CLS_CONST LaneGroupId kDefaultGroup = 0;
  FullPtr<ReorganizeTask> reorg_;
  Client client_;
  size_t poll_block_ = 0;
  size_t poll_thread_ = 0;

 public:
  Server() = default;

  CHI_BEGIN(Create)
  /** Construct eternia_core */
  void Create(CreateTask *task, RunContext &rctx) {
    CreateTaskParams params = task->GetParams();
    CreateLaneGroup(kDefaultGroup, 1, QUEUE_LOW_LATENCY);
    client_.Init(id_);
    // Initialize queues on each GPU
    for (int gpu_id = 0; gpu_id < CHI_CLIENT->ngpu_; ++gpu_id) {
      hshm::GpuApi::SetDevice(gpu_id);
      auto *gcache =
          hshm::GpuApi::Malloc<FullPtr<GpuCache>>(sizeof(FullPtr<GpuCache>));
      InitEterniaRuntime<<<1, 1>>>(params.qcount_, params.qdepth_, gcache,
                                   params.hermes_);
      hshm::GpuApi::Synchronize();
      client_.gcache_[gpu_id] = *gcache;
      hshm::GpuApi::Free(gcache);
    }
    // Copy queues to CreateTaskParams
    memcpy(params.gcache_, client_.gcache_,
           sizeof(FullPtr<GpuCache>) * CHI_CLIENT->ngpu_);
    task->SetParams(params);
    // Get the dimensions of the polling function
    poll_block_ = params.qcount_ / 1024;
    poll_thread_ = params.qcount_ % 1024;
    if (poll_block_ == 0) {
      poll_block_ = 1;
    }
    if (poll_thread_ == 0) {
      poll_thread_ = 1024;
    }
    // Begin polling the queues on this container
    reorg_ = client_.AsyncReorganize(HSHM_MCTX, DomainQuery::GetLocalHash(0));
  }
  void MonitorCreate(MonitorModeId mode, CreateTask *task, RunContext &rctx) {}
  CHI_END(Create)

  /** Route a task to a lane */
  Lane *MapTaskToLane(const Task *task) override {
    // Route tasks to lanes based on their properties
    // E.g., a strongly consistent filesystem could map tasks to a lane
    // by the hash of an absolute filename path.
    return GetLaneByHash(kDefaultGroup, task->prio_, 0);
  }

  CHI_BEGIN(Destroy)
  /** Destroy eternia_core */
  void Destroy(DestroyTask *task, RunContext &rctx) {}
  void MonitorDestroy(MonitorModeId mode, DestroyTask *task, RunContext &rctx) {
  }
  CHI_END(Destroy)

  CHI_BEGIN(Reorganize)
  /** The Reorganize method */
  void Reorganize(ReorganizeTask *task, RunContext &rctx) {
    for (int gpu_id = 0; gpu_id < CHI_CLIENT->ngpu_; ++gpu_id) {
      FullPtr<GpuCache> &gcache = client_.gcache_[gpu_id];
      hshm::GpuApi::SetDevice(gpu_id);
      PollCpuQueue<<<1, 1>>>(gcache.shm_);
      PollGpuQueues<<<poll_block_, poll_thread_>>>(gcache.shm_);
    }
    for (int gpu_id = 0; gpu_id < CHI_CLIENT->ngpu_; ++gpu_id) {
      hshm::GpuApi::SetDevice(gpu_id);
      hshm::GpuApi::Synchronize();
    }
  }
  void MonitorReorganize(MonitorModeId mode, ReorganizeTask *task,
                         RunContext &rctx) {
    switch (mode) {
      case MonitorMode::kReplicaAgg: {
        std::vector<FullPtr<Task>> &replicas = *rctx.replicas_;
      }
    }
  }
  CHI_END(Reorganize)

  CHI_AUTOGEN_METHODS  // keep at class bottom
      public:
#include "eternia_core/eternia_core_lib_exec.h"
};

}  // namespace eternia

CHI_TASK_CC(eternia::Server, "eternia_core");
