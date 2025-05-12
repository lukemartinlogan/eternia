#ifndef ETERNIA_TRANSACTION_H
#define ETERNIA_TRANSACTION_H

#include "constants.h"
#include "vector.h"

namespace eternia {

#define ET_READ BIT_OPT(u32, 0)
#define ET_WRITE BIT_OPT(u32, 1)
#define ET_RW BIT_OPT(u32, 2)

typedef u32 IoFlags;

class Transaction {
 public:
  HSHM_GPU_FUN Transaction() = default;
  HSHM_GPU_FUN ~Transaction() = default;

  template <typename TxT>
  HSHM_GPU_FUN void Prefetch(TxT &tx) {
    // Mark unused pages
    for (const PageRegion &region : tx.GetTouchedPages()) {
      tx.vec_.SuggestScore(region, 0.0);
    }
    // Mark expected pages
    for (const PageRegion &region : tx.GetFuturePages()) {
      tx.vec_.SuggestScore(region, 1.0);
    }
    // Process faults and evictions from marks
    for (const PageRegion &region : tx.GetTouchedPages()) {
      tx.vec_.SolidifyScore(region, 0.0);
    }
    for (const PageRegion &region : tx.GetFuturePages()) {
      tx.vec_.SolidifyScore(region, 1.0);
    }
  }
};

};  // namespace eternia

#endif