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

#ifndef CHI_eternia_core_H_
#define CHI_eternia_core_H_

#include <hermes_shm/constants/macros.h>

#include "eternia_core_tasks.h"

namespace eternia {

/** Create eternia_core requests */
class Client : public ModuleClient {
 public:
  FullPtr<GpuCache> gcache_[HSHM_MAX_GPUS];

 public:
  /** Default constructor */
  HSHM_INLINE_CROSS_FUN
  Client() {
    for (int i = 0; i < HSHM_MAX_GPUS; ++i) {
      gcache_[i] = FullPtr<GpuCache>::GetNull();
    }
  }

  /** Destructor */
  HSHM_INLINE_CROSS_FUN
  ~Client() = default;

  CHI_BEGIN(Create)
  /** Create a pool */
  HSHM_INLINE_CROSS_FUN
  void Create(const hipc::MemContext &mctx, const DomainQuery &dom_query,
              const DomainQuery &affinity, const chi::string &pool_name,
              const CreateContext &ctx = CreateContext()) {
    FullPtr<CreateTask> task =
        AsyncCreate(mctx, dom_query, affinity, pool_name, ctx);
    task->Wait();
    Init(task->ctx_.id_);
    CreateTaskParams params = task->GetParams();
    memcpy(gcache_, params.gcache_, sizeof(FullPtr<GpuCache>) * HSHM_MAX_GPUS);
    CHI_CLIENT->DelTask(mctx, task);
  }
  CHI_TASK_METHODS(Create);
  CHI_END(Create)

  CHI_BEGIN(Destroy)
  /** Destroy pool + queue */
  HSHM_INLINE_CROSS_FUN
  void Destroy(const hipc::MemContext &mctx, const DomainQuery &dom_query) {
    FullPtr<DestroyTask> task = AsyncDestroy(mctx, dom_query, id_);
    task->Wait();
    CHI_CLIENT->DelTask(mctx, task);
  }
  CHI_TASK_METHODS(Destroy)
  CHI_END(Destroy)

  CHI_BEGIN(Reorganize)
  /** Reorganize task */
  void Reorganize(const hipc::MemContext &mctx, const DomainQuery &dom_query) {
    FullPtr<ReorganizeTask> task = AsyncReorganize(mctx, dom_query);
    task->Wait();
    CHI_CLIENT->DelTask(mctx, task);
  }
  CHI_TASK_METHODS(Reorganize);
  CHI_END(Reorganize)

  CHI_AUTOGEN_METHODS  // keep at class bottom
};

}  // namespace eternia

#define ETERNIA_CLIENT hshm::Singleton<eternia::Client>::GetInstance()

static inline void ETERNIA_INIT() {
  HERMES_INIT();
  ETERNIA_CLIENT->Create(
      HSHM_MCTX,
      chi::DomainQuery::GetDirectHash(chi::SubDomain::kGlobalContainers, 0),
      chi::DomainQuery::GetGlobalBcast(), "eternia_core");
}

#endif  // CHI_eternia_core_H_
