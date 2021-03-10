#ifndef DNE_BINARY_H
#define DNE_BINARY_H

#include <Eigen/Core>

void binary_mult_self(const Eigen::MatrixXd &B,
                      Eigen::MatrixXd &outB_Bt);

void binary_mult(const Eigen::MatrixXd &A,
                 const Eigen::MatrixXd &B,
                 Eigen::MatrixXd &outC);

void binary_mult512_self(const Eigen::MatrixXd &A,
                 Eigen::MatrixXd &outC);

template<class T>
inline void Log(const __m256i & value);

static double F_norm_pow2(const Eigen::MatrixXd &in){
  return (in * in.transpose()).trace();
}

#endif //DNE_BINARY_H
