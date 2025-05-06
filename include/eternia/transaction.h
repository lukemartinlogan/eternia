#include "constants.h"
#include "vector.h"

namespace eternia {

class Transaction {
  et::Comm agg_;

  Transaction(et::Comm agg);
  ~Transaction();
};

};  // namespace eternia