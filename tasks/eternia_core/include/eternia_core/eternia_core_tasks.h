//
// Created by lukemartinlogan on 8/11/23.
//

#ifndef CHI_TASKS_TASK_TEMPL_INCLUDE_eternia_core_eternia_core_TASKS_H_
#define CHI_TASKS_TASK_TEMPL_INCLUDE_eternia_core_eternia_core_TASKS_H_

#include "chimaera/chimaera_namespace.h"
#include "eternia/libgpu.h"

namespace eternia {

#include "eternia_core_methods.h"
CHI_NAMESPACE_INIT

CHI_BEGIN(Create)
/** A task to create eternia_core */
struct CreateTaskParams {
  CLS_CONST char *lib_name_ = "eternia_eternia_core";
  IN int qcount_ = 1024;
  IN int qdepth_ = 64;
  IN hermes::Client hermes_;
  OUT hipc::FullPtr<GpuCache> gcache_[HSHM_MAX_GPUS];

  HSHM_INLINE_CROSS_FUN
  CreateTaskParams() = default;

  HSHM_INLINE_CROSS_FUN
  CreateTaskParams(const hipc::CtxAllocator<CHI_ALLOC_T> &alloc,
                   int qcount = 1024, int qdepth = 64) {
    hermes_.Init(HERMES_CONF->mdm_.pool_id_);
    qcount_ = qcount;
    qdepth_ = qdepth;
  }

  template <typename Ar>
  HSHM_INLINE_CROSS_FUN void serialize(Ar &ar) {
    ar(hermes_, qcount_, qdepth_, gcache_);
  }
};
typedef chi::Admin::CreatePoolBaseTask<CreateTaskParams> CreateTask;
CHI_END(Create)

CHI_BEGIN(Destroy)
/** A task to destroy eternia_core */
typedef chi::Admin::DestroyContainerTask DestroyTask;
CHI_END(Destroy)

CHI_BEGIN(Reorganize)
/** The ReorganizeTask task */
struct ReorganizeTask : public Task, TaskFlags<TF_SRL_SYM> {
  /** SHM default constructor */
  HSHM_INLINE explicit ReorganizeTask(
      const hipc::CtxAllocator<CHI_ALLOC_T> &alloc)
      : Task(alloc) {}

  /** Emplace constructor */
  HSHM_INLINE explicit ReorganizeTask(
      const hipc::CtxAllocator<CHI_ALLOC_T> &alloc, const TaskNode &task_node,
      const PoolId &pool_id, const DomainQuery &dom_query)
      : Task(alloc) {
    // Initialize task
    task_node_ = task_node;
    prio_ = TaskPrioOpt::kHighLatency;
    pool_ = pool_id;
    method_ = Method::kReorganize;
    task_flags_.SetBits(TASK_LONG_RUNNING);
    dom_query_ = dom_query;

    // Custom
    SetPeriodUs(100);
  }

  /** Duplicate message */
  void CopyStart(const ReorganizeTask &other, bool deep) {}

  /** (De)serialize message call */
  template <typename Ar>
  void SerializeStart(Ar &ar) {}

  /** (De)serialize message return */
  template <typename Ar>
  void SerializeEnd(Ar &ar) {}
};
CHI_END(Reorganize);

CHI_AUTOGEN_METHODS  // keep at class bottom

}  // namespace eternia

#endif  // CHI_TASKS_TASK_TEMPL_INCLUDE_eternia_core_eternia_core_TASKS_H_
