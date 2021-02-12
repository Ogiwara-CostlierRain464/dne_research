#ifndef DNE_BMP_H
#define DNE_BMP_H

#include <Eigen/Core>
using Eigen::MatrixXf;
using Eigen::Matrix;
typedef Matrix<unsigned int,Eigen::Dynamic,Eigen::Dynamic> MatrixXi;

unsigned int concatenate(const float* array);
MatrixXi concatenate(MatrixXf A);
float* deconcatenate(unsigned int x);
MatrixXf deconcatenate(MatrixXi A);

MatrixXf BMP(MatrixXf A,MatrixXf B);

#endif //DNE_BMP_H
