#ifndef CHI_ETERNIA_CORE_METHODS_H_
#define CHI_ETERNIA_CORE_METHODS_H_

/** The set of methods in the admin task */
struct Method : public chi::TaskMethod {
  TASK_METHOD_T kReorganize = 10;
  TASK_METHOD_T kCount = 11;
};

#endif  // CHI_ETERNIA_CORE_METHODS_H_