#ifndef ETERNIA_TRANSACTIONS_SEQUENTIAL_H
#define ETERNIA_TRANSACTIONS_SEQUENTIAL_H

#include "eternia/constants.h"
#include "eternia/vector.h"

namespace eternia {

template <typename VecT>
class SeqTxIterator {
 private:
  VecT &vec_;
  size_t off_;
  size_t size_;
  size_t tail_;

 public:
  SeqTxIterator(VecT &vec, size_t off, size_t tail, size_t size)
      : vec_(vec), off_(off), size_(size), tail_(tail) {}

  SeqTxIterator &operator++() {
    tail_ += vec_.ctx_.tcache_page_size_;
    return *this;
  }

  bool operator!=(const SeqTxIterator &other) const {
    return tail_ != other.tail_;
  }

  MemTask operator*() const {
    MemTask task;
    task.tag_id_ = vec_.tag_id_;
    task.off_ = off_ + tail_ * vec_.ctx_.tcache_page_size_;
    task.size_ = size_;
    return task;
  }
};

template <typename T, typename VecT>
class SeqTx {
 public:
  using iterator = SeqTxIterator<VecT>;
  VecT &vec_;
  size_t off_;
  size_t size_;
  size_t head_ = 0;        // Prefetch head
  size_t tail_ = 0;        // Prefetch tail
  size_t lookahead_ = 16;  // Prefetch lookahead

  SeqTx(VecT &vec, size_t off, size_t size) : off_(off), size_(size) {}

  T &operator[](size_t i) {
    T *val;
    if (vec_.FindValInTcache(i + off_, val)) {
      return *val;
    }
    tail_ = i;
    Prefetch();
    if (vec_.FindValInTcache(i + off_, val)) {
      return *val;
    }
    return *val;
  }

  iterator GetTouchedPages() {
    return iterator(vec_, off_, head_, tail_ - head_);
  }

  iterator GetFuturePages() { return iterator(vec_, off_, tail_, lookahead_); }

  void Prefetch() {
    // Evict unused pages
    for (MemTask &task : vec_.GetTouchedPages()) {
      vec_.Evict(task);
    }
    // Prefetch next pages
    for (MemTask &task : vec_.GetFuturePages()) {
      vec_.Prefetch(task);
    }
    // Ensure prefetch doesn't re-evict
    head_ = tail_;
  }
};
}  // namespace eternia

#endif  // ETERNIA_TRANSACTIONS_SEQUENTIAL_H