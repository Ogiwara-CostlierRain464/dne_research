#ifndef DNE_PARAM_H
#define DNE_PARAM_H

#include <gflags/gflags.h>


namespace fLD{
  extern double FLAGS_ab;
  extern double FLAGS_train_ratio;
  extern double FLAGS_tau;
  extern double FLAGS_lambda;
  extern double FLAGS_mu;
  extern double FLAGS_rho;
}

using fLD::FLAGS_ab;
using fLD::FLAGS_train_ratio;
using fLD::FLAGS_tau;
using fLD::FLAGS_lambda;
using fLD::FLAGS_mu;
using fLD::FLAGS_rho;

namespace fLS{
  extern clstring& FLAGS_dataset;
}

using fLS::FLAGS_dataset;

namespace fLU{
  extern gflags::uint32 FLAGS_m;
  extern gflags::uint32 FLAGS_T_in;
  extern gflags::uint32 FLAGS_T_out;
}

using fLU::FLAGS_m;
using fLU::FLAGS_T_in;
using fLU::FLAGS_T_out;







#endif //DNE_PARAM_H
