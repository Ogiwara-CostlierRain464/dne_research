//
// Created by ogiwara on 1/27/2021.
//
#include <iostream>
#include <fstream>
#include <random>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

class Save: public ::testing::Test{};

namespace Eigen {
// https://stackoverflow.com/a/25389481/11927397
    template<class Matrix>
    inline void write_binary(const std::string& filename, const Matrix& matrix){
        std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
        if(out.is_open()) {
            typename Matrix::Index rows=matrix.rows(), cols=matrix.cols();
            out.write(reinterpret_cast<char*>(&rows), sizeof(typename Matrix::Index));
            out.write(reinterpret_cast<char*>(&cols), sizeof(typename Matrix::Index));
            out.write(reinterpret_cast<const char*>(matrix.data()), rows*cols*static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)) );
            out.close();
        }
        else {
            std::cout << "Can not write to file: " << filename << std::endl;
        }
    }

    template<class Matrix>
    inline void read_binary(const std::string& filename, Matrix& matrix){
        std::ifstream in(filename, std::ios::in | std::ios::binary);
        if (in.is_open()) {
            typename Matrix::Index rows=0, cols=0;
            in.read(reinterpret_cast<char*>(&rows),sizeof(typename Matrix::Index));
            in.read(reinterpret_cast<char*>(&cols),sizeof(typename Matrix::Index));
            matrix.resize(rows, cols);
            in.read(reinterpret_cast<char*>(matrix.data()), rows*cols*static_cast<typename Matrix::Index>(sizeof(typename Matrix::Scalar)) );
            in.close();
        }
        else {
            std::cout << "Can not open binary matrix file: " << filename << std::endl;
        }
    }

// https://scicomp.stackexchange.com/a/21438
    template <class SparseMatrix>
    inline void write_binary_sparse(const std::string& filename, const SparseMatrix& matrix) {
        assert(matrix.isCompressed() == true);
        std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
        if(out.is_open())
        {
            typename SparseMatrix::Index rows, cols, nnzs, outS, innS;
            rows = matrix.rows()     ;
            cols = matrix.cols()     ;
            nnzs = matrix.nonZeros() ;
            outS = matrix.outerSize();
            innS = matrix.innerSize();

            out.write(reinterpret_cast<char*>(&rows), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&cols), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&nnzs), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&outS), sizeof(typename SparseMatrix::Index));
            out.write(reinterpret_cast<char*>(&innS), sizeof(typename SparseMatrix::Index));

            typename SparseMatrix::Index sizeIndexS = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::StorageIndex));
            typename SparseMatrix::Index sizeScalar = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Scalar      ));
            out.write(reinterpret_cast<const char*>(matrix.valuePtr()),       sizeScalar * nnzs);
            out.write(reinterpret_cast<const char*>(matrix.outerIndexPtr()),  sizeIndexS  * outS);
            out.write(reinterpret_cast<const char*>(matrix.innerIndexPtr()),  sizeIndexS  * nnzs);

            out.close();
        }
        else {
            std::cout << "Can not write to file: " << filename << std::endl;
        }
    }

    template <class SparseMatrix>
    inline void read_binary_sparse(const std::string& filename, SparseMatrix& matrix) {
        std::ifstream in(filename, std::ios::binary | std::ios::in);
        if(in.is_open()) {
            typename SparseMatrix::Index rows, cols, nnz, inSz, outSz;
            typename SparseMatrix::Index sizeScalar = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Scalar      ));
            typename SparseMatrix::Index sizeIndex  = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::Index       ));
            typename SparseMatrix::Index sizeIndexS = static_cast<typename SparseMatrix::Index>(sizeof(typename SparseMatrix::StorageIndex));
            std::cout << sizeScalar << " " << sizeIndex << std::endl;
            in.read(reinterpret_cast<char*>(&rows ), sizeIndex);
            in.read(reinterpret_cast<char*>(&cols ), sizeIndex);
            in.read(reinterpret_cast<char*>(&nnz  ), sizeIndex);
            in.read(reinterpret_cast<char*>(&outSz), sizeIndex);
            in.read(reinterpret_cast<char*>(&inSz ), sizeIndex);

            matrix.resize(rows, cols);
            matrix.makeCompressed();
            matrix.resizeNonZeros(nnz);

            in.read(reinterpret_cast<char*>(matrix.valuePtr())     , sizeScalar * nnz  );
            in.read(reinterpret_cast<char*>(matrix.outerIndexPtr()), sizeIndexS * outSz);
            in.read(reinterpret_cast<char*>(matrix.innerIndexPtr()), sizeIndexS * nnz );

            matrix.finalize();
            in.close();
        } // file is open
        else {
            std::cout << "Can not open binary sparse matrix file: " << filename << std::endl;
        }
    }
} // Eigen::

TEST(Save, txt){
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_MxN;

    // https://stackoverflow.com/a/56459780/11927397
    std::random_device rd;
    std::mt19937 gen(rd());  //here you could set the seed, but std::random_device already does that
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    Matrix_MxN J = Matrix_MxN::NullaryExpr(10,5,[&](){return dis(gen);});
    Eigen::write_binary("dense.bin",J);
    std::cout << "\n original \n" << J << std::endl;
    Matrix_MxN J_copy;
    Eigen::read_binary("dense.bin",J_copy);
    std::cout << "\n copy \n" << J_copy << std::endl;

    return;

    int rows, cols;
    rows = cols = 6;
    Eigen::SparseMatrix<double> A(rows,cols), B;
    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trp;

    trp.push_back(Trip(0, 0, dis(gen)));
    trp.push_back(Trip(1, 1, dis(gen)));
    trp.push_back(Trip(2, 2, dis(gen)));
    trp.push_back(Trip(3, 3, dis(gen)));
    trp.push_back(Trip(4, 4, dis(gen)));
    trp.push_back(Trip(5, 5, dis(gen)));
    trp.push_back(Trip(2, 4, dis(gen)));
    trp.push_back(Trip(3, 1, dis(gen)));

    A.setFromTriplets(trp.begin(), trp.end());
    std::cout << A.nonZeros() << std::endl;   // Prints 8
    std::cout << A.size() << std::endl;       // Prints 36
    std::cout << Eigen::MatrixXd(A) << std::endl;              // Prints the matrix along with the sparse matrix stuff

    Eigen::write_binary_sparse("sparse.bin", A);
    Eigen::read_binary_sparse("sparse.bin", B);

    std::cout << B.nonZeros() << std::endl;   // Prints 8
    std::cout << B.size() << std::endl;       // Prints 36
    std::cout << Eigen::MatrixXd(B) << std::endl;              // Prints the reconstructed matrix along with the sparse matrix stuff

}

