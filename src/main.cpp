#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

void at_runtime(){
  MatrixXd m = MatrixXd::Random(3,3);
  m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
  cout << "m =" << endl << m << endl;
  VectorXd v(3);
  v << 1, 2, 3;
  cout << "m * v =" << endl << m * v << endl;
}

void at_compile_time(){
  Matrix3d m = Matrix3d::Random();
  m = (m + Matrix3d::Constant(1.2)) * 50;
  cout << "m =" << endl << m << endl;
  Vector3d v(1,2,3);
  cout << "m * v =" << endl << m * v << endl;
}

int main(){
  at_runtime();
  at_compile_time();
}