#ifndef DNE_BINARY_H
#define DNE_BINARY_H

#include <Eigen/Core>

void binary_mult(const Eigen::MatrixXd &A,
                 const Eigen::MatrixXd &B,
                 Eigen::MatrixXd &outC);

void binary_mult512(const Eigen::MatrixXd &A,
                 const Eigen::MatrixXd &B,
                 Eigen::MatrixXd &outC);

#endif //DNE_BINARY_H
