#ifndef ETERNIA_TRANSACTIONS_SEQUENTIAL_H
#define ETERNIA_TRANSACTIONS_SEQUENTIAL_H

#include "eternia/constants.h"
#include "eternia/transaction.h"
#include "eternia/vector.h"

namespace eternia {

template <typename VecT>
class SeqTxIterator {
 private:
  VecT &vec_;
  size_t off_;
  size_t size_;
  size_t pos_;

 public:
  HSHM_GPU_FUN
  SeqTxIterator(VecT &vec, size_t off, size_t size, size_t pos = 0)
      : vec_(vec), off_(off), size_(size), pos_(pos) {}

  HSHM_GPU_FUN
  SeqTxIterator &operator++() {
    size_t rem = pos_ % vec_.ctx_.tcache_page_size_;
    pos_ += vec_.ctx_.tcache_page_size_ - rem;
    return *this;
  }

  HSHM_GPU_FUN
  bool operator!=(const SeqTxIterator &other) const {
    return pos_ != other.pos_;
  }

  HSHM_GPU_FUN
  bool operator==(const SeqTxIterator &other) const {
    return pos_ == other.pos_;
  }

  HSHM_GPU_FUN
  SeqTxIterator begin() const { return iterator(vec_, off_, size_, 0); }

  HSHM_GPU_FUN
  SeqTxIterator end() const { return iterator(vec_, off_, size_, size_); }

  HSHM_GPU_FUN
  PageRegion operator*() const {
    return PageRegion(off_ + pos_, vec_.ctx_.tcache_page_size_, false);
  }
};

template <typename T, typename VecT>
class SeqTx : public Transaction {
 public:
  using iterator = SeqTxIterator<VecT>;
  VecT *vec_;
  size_t off_;             // Offet from beginning of vector
  size_t size_;            // Amount of data in transaction
  size_t head_ = 0;        // Prefetch head
  size_t tail_ = 0;        // Prefetch tail
  size_t lookahead_ = 16;  // Prefetch lookahead
  size_t interval_ = 8;    // Prefetch interval

  HSHM_GPU_FUN
  SeqTx(VecT &vec, size_t off, size_t size)
      : vec_(vec), off_(off), size_(size) {}

  HSHM_GPU_FUN
  T &operator[](size_t i) {
    T *val;
    if (vec_->page_bnds_ % interval_ == 0) {
      Prefetch();
    }
    if (vec_->FindValInTcache(i + off_, val)) {
      return *val;
    }
    tail_ = i;
    Prefetch();
    if (vec_->FindValInTcache(i + off_, val)) {
      return *val;
    }
    return *val;
  }

  HSHM_GPU_FUN
  iterator GetTouchedPages() {
    return iterator(*vec_, off_, head_, tail_ - head_);
  }

  HSHM_GPU_FUN
  iterator GetFuturePages() { return iterator(*vec_, off_, tail_, lookahead_); }

  HSHM_GPU_FUN
  void Prefetch() {
    Prefetch(*this);
    head_ = tail_;
  }
};
}  // namespace eternia

#endif  // ETERNIA_TRANSACTIONS_SEQUENTIAL_H