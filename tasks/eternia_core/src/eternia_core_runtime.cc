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

#include "chimaera/api/chimaera_runtime.h"
#include "chimaera/monitor/monitor.h"
#include "chimaera_admin/chimaera_admin_client.h"
#include "eternia_core/eternia_core_client.h"

namespace chi::eternia_core {

struct MemTask {};

struct EterniaMq {
  chi::data::ipc::vector<chi::data::ipc::mpsc_queue<MemTask>> gpu_queues_;
  chi::ipc::mpsc_queue<MemTask> cpu_queue_;

  EterniaMq(int count, int depth)
      : gpu_queues_(count, depth), cpu_queue_(depth) {}
};

HSHM_GPU_KERNEL void InitEterniaRuntime(int qcount, int qdepth,
                                        FullPtr<EterniaMq> *et) {
  hipc::ScopedTlsAllocator<CHI_DATA_ALLOC_T> tls(CHI_CLIENT->data_alloc_);
  *et = tls.alloc_->NewObjLocal<EterniaMq>(tls.alloc_.ctx_, qcount, qdepth);
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

HSHM_GPU_KERNEL void EterniaRuntime(EterniaMq mq) {
  size_t tid = hshm::GpuApi::GetGlobalThreadId();
  if (tid == 0) {
    PollEterniaQueue(mq.cpu_queue_);
  } else {
    tid -= 1;
    PollEterniaQueue(mq.gpu_queues_[tid]);
  }
}

class Server : public Module {
 public:
  CLS_CONST LaneGroupId kDefaultGroup = 0;

 public:
  Server() = default;

  CHI_BEGIN(Create)
  /** Construct eternia_core */
  void Create(CreateTask *task, RunContext &rctx) {
    // Create a set of lanes for holding tasks
    CreateLaneGroup(kDefaultGroup, 1, QUEUE_LOW_LATENCY);
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
  void Reorganize(ReorganizeTask *task, RunContext &rctx) {}
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

}  // namespace chi::eternia_core

CHI_TASK_CC(chi::eternia_core::Server, "eternia_core");
