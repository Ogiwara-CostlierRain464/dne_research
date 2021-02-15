#include "binary.h"
#include "../third_party/libpopcnt/libpopcnt.h"
#include <immintrin.h>
#include <iostream>

using Eigen::MatrixXd;

namespace {

inline __attribute__((__always_inline__)) uint64_t h_dis(const __m256i a, const __m256i b){
  __m256i xor_ = _mm256_xor_epi64(a, b);
  return popcnt(&xor_, 256/8);
}

  inline __attribute__((__always_inline__)) uint64_t h_dis512(const __m512i a, const __m512i b){
    __m512i xor_ = _mm512_xor_epi64(a, b);
    return popcnt(&xor_, 512/8);
  }
}

#define BIT_SIZE 256
#define BIT_UNIT_TYPE __m256i

void binary_mult(const Eigen::MatrixXd &A,
                 const Eigen::MatrixXd &B,
                 Eigen::MatrixXd &outC) {


  auto A_row = A.rows();
  auto A_col = A.cols();
  auto B_row = B.rows();
  auto B_col = B.cols();

  // AのcolとBのrowを増やす必要がある
  auto A_col_inc = A_col % 256 == 0 ? A_col : (A_col - (A_col % 256)) + 256;
  auto B_row_inc = B_row % 256 == 0 ? B_row : (B_row - (B_row % 256)) + 256;

  assert(A_col_inc % 256 == 0);
  assert(B_row_inc % 256 == 0);

  auto A_arr = new __m256i[ A_row * (A_col_inc / 256)];

#pragma omp parallel for collapse(2)
  for(size_t i = 0; i < A_row; ++i){
    for(size_t j = 0; j < A_col_inc; j+=256){
      // 64個の数字ごとに格納
      // column orderであることに留意

      uint64_t part[4] = {};

      for(size_t p = 0; p < 4; ++p){
        for(size_t k = p * 64; k < (p+1)*64; ++k){
          unsigned int sign = j+k >= A_col ? 0 : (*(A.data() + i + (k+j) * A_row) > 0);
          part[p] = (part[p] << 1) | sign;
        }
      }

      A_arr[i * (A_col_inc/256) + (j/256)] = _mm256_set_epi64x(part[0], part[1], part[2], part[3]);
    }
  }

  // tmp

  outC = MatrixXd(A_row, A_row);

#pragma omp parallel for collapse(2)
  for(size_t i = 0; i < A_row; ++i){
    for(size_t j = 0; j < A_row; ++j){
      uint64_t h_dis_ij = 0;
      for(size_t part = 0; part < ( A_col_inc / 256 );  ++part ){
        h_dis_ij += h_dis(A_arr[i * (A_col_inc/256) + part], A_arr[ j * (A_col_inc/256) + part ]);
      }

      int C_ = A_col;
      int h_dis_ij_ = h_dis_ij;

      outC.coeffRef(i,j) = C_ - 2 * h_dis_ij_;
    }
  }
}

void binary_mult512(const Eigen::MatrixXd &A,
                 const Eigen::MatrixXd &B,
                 Eigen::MatrixXd &outC) {


  auto A_row = A.rows();
  auto A_col = A.cols();
  auto B_row = B.rows();
  auto B_col = B.cols();

  // AのcolとBのrowを増やす必要がある
  auto A_col_inc = A_col % 512 == 0 ? A_col : (A_col - (A_col % 512)) + 512;
  auto B_row_inc = B_row % 512 == 0 ? B_row : (B_row - (B_row % 512)) + 512;

  assert(A_col_inc % 512 == 0);
  assert(B_row_inc % 512 == 0);

  auto A_arr = new __m512i[ A_row * (A_col_inc / 512)];

#pragma omp parallel for collapse(2)
  for(size_t i = 0; i < A_row; ++i){
    for(size_t j = 0; j < A_col_inc; j+=512){
      // 64個の数字ごとに格納
      // column orderであることに留意

      uint64_t part[8] = {};

      for(size_t p = 0; p < 8; ++p){
        for(size_t k = p * 64; k < (p+1)*64; ++k){
          unsigned int sign = j+k >= A_col ? 0 : (*(A.data() + i + (k+j) * A_row) > 0);
          part[p] = (part[p] << 1) | sign;
        }
      }

      A_arr[i * (A_col_inc/512) + (j/512)] = _mm512_set_epi64(part[0], part[1], part[2], part[3],
                                                              part[4], part[5], part[6], part[7]);
    }
  }

  // tmp

  outC = MatrixXd(A_row, A_row);

#pragma omp parallel for collapse(2)
  for(size_t i = 0; i < A_row; ++i){
    for(size_t j = 0; j < A_row; ++j){
      uint64_t h_dis_ij = 0;
      for(size_t part = 0; part < ( A_col_inc / 512 );  ++part ){
        h_dis_ij += h_dis512(A_arr[i * (A_col_inc/512) + part], A_arr[ j * (A_col_inc/512) + part ]);
      }

      int C_ = A_col;
      int h_dis_ij_ = h_dis_ij;

      outC.coeffRef(i,j) = C_ - 2 * h_dis_ij_;
    }
  }
}
