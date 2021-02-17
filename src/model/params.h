#ifndef DNE_PARAMS_H
#define DNE_PARAMS_H

#include <cstddef>

struct Params{
  Params(
    const size_t &m,
    const size_t &tIn,
    const size_t &tOut,
    double tau,
    double lambda,
    double mu,
    double rho,
    size_t seed,
    bool check_loss) :
    m(m), T_in(tIn), T_out(tOut),
    tau(tau), lambda(lambda), mu(mu),
    rho(rho), seed(seed), check_loss(check_loss) {}

  size_t m;
  size_t T_in;
  size_t T_out;
  double tau;
  double lambda;
  double mu;
  double rho;
  size_t seed;
  bool check_loss;
};

#endif //DNE_PARAMS_H
