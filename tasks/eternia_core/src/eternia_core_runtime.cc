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

#include <hermes_shm/util/gpu_api.h>

#include "chimaera/api/chimaera_runtime.h"
#include "chimaera/monitor/monitor.h"
#include "chimaera_admin/chimaera_admin_client.h"
#include "eternia_core/eternia_core_client.h"
#include "eternia_core/eternia_core_tasks.h"

namespace eternia {

HSHM_GPU_KERNEL void InitEterniaRuntime(int qcount, int qdepth,
                                        FullPtr<EterniaMq> *et_mq) {
  hipc::ScopedTlsAllocator<CHI_DATA_ALLOC_T> tls(CHI_CLIENT->data_alloc_);
  *et_mq = tls.alloc_->NewObjLocal<EterniaMq>(tls.alloc_.ctx_, qcount, qdepth);
}

template <typename QueueT>
HSHM_GPU_FUN void PollEterniaQueue(QueueT &queue) {
  size_t count = queue.size();
  for (size_t i = 0; i < count; ++i) {
    MemTask task;
    if (queue.pop(task).IsNull()) {
      break;
    }
    // Process the task
  }
}

HSHM_GPU_KERNEL void PollCpuQueue(hipc::Pointer mq_p) {
  hipc::FullPtr<EterniaMq> mq(mq_p);
  size_t tid = hshm::GpuApi::GetGlobalThreadId();
  if (tid == 0) {
    PollEterniaQueue(mq->cpu_queue_);
  }
}

HSHM_GPU_KERNEL void PollGpuQueues(hipc::Pointer mq_p) {
  hipc::FullPtr<EterniaMq> mq(mq_p);
  size_t tid = hshm::GpuApi::GetGlobalThreadId();
  if (tid < mq->gpu_queues_.size()) {
    PollEterniaQueue(mq->gpu_queues_[tid]);
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
      auto *et_mq =
          hshm::GpuApi::Malloc<FullPtr<EterniaMq>>(sizeof(FullPtr<EterniaMq>));
      InitEterniaRuntime<<<1, 1>>>(params.qcount_, params.qdepth_, et_mq);
      hshm::GpuApi::Synchronize();
      client_.et_mq_[gpu_id] = *et_mq;
      hshm::GpuApi::Free(et_mq);
    }
    // Copy queues to CreateTaskParams
    memcpy(params.et_mq_, client_.et_mq_,
           sizeof(FullPtr<EterniaMq>) * CHI_CLIENT->ngpu_);
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
      FullPtr<EterniaMq> &et_mq = client_.et_mq_[gpu_id];
      hshm::GpuApi::SetDevice(gpu_id);
      PollCpuQueue<<<1, 1>>>(et_mq.shm_);
      PollGpuQueues<<<poll_block_, poll_thread_>>>(et_mq.shm_);
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
