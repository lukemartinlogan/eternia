#include "constants.h"



namespace eternia {

class Transaction {
  et::Comm agg_;

  Transaction(et::Comm agg);
  ~Transaction();

  int& operator[](size_t off);
  const int& operator[](size_t off) const;
  void Log(int off);
  int Prefetch(int lookahead, MemTask *regions);
  void Fault(int off);
};


template <typename T, typename VecT> class SeqTx {
  VecT &vec_;
  size_t off_;
  size_t size_;

  SeqTx(VecT &vec, size_t off, size_t size) : off_(off), size_(size) {}

  T &operator[](size_t i) {
    
    return vec_[i + off_];
  }
  
};

};  // namespace eternia