#include "binary.h"
#include "../third_party/libpopcnt/libpopcnt.h"
#include <immintrin.h>
#include <avx2intrin.h>
#include <avx512vlintrin.h>
#include <iostream>

using Eigen::MatrixXd;

namespace {

inline __attribute__((__always_inline__)) uint64_t h_dis(const __m256i a, const __m256i b){
  auto xor_ = (__m256i) ((__v4du) a ^ (__v4du) b);
  return popcnt(&xor_, 256/8);
}

  inline __attribute__((__always_inline__)) uint64_t h_dis512(const __m512i a, const __m512i b){
    __m512i xor_ = _mm512_xor_epi64(a, b);
    return popcnt(&xor_, 512/8);
  }
}

void binary_mult(const Eigen::MatrixXd &A,
                 const Eigen::MatrixXd &B,
                 Eigen::MatrixXd &outC) {


  auto A_row = A.rows();
  auto A_col = A.cols();
  auto B_row = B.rows();
  auto B_col = B.cols();

  assert(A_col == B_row);

  // AのcolとBのrowを増やす必要がある
  auto A_col_inc = A_col % 256 == 0 ? A_col : (A_col - (A_col % 256)) + 256;
  auto B_row_inc = B_row % 256 == 0 ? B_row : (B_row - (B_row % 256)) + 256;

  assert(A_col_inc % 256 == 0);
  assert(B_row_inc % 256 == 0);

  auto A_arr = new (std::align_val_t{64}) __m256i[ A_row * (A_col_inc / 256)];

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

  auto B_arr = new (std::align_val_t{64}) __m256i[ B_col * (B_row_inc / 256)];

  #pragma omp parallel for collapse(2)
  for(size_t i = 0; i < B_row_inc; i+=256){
    for(size_t j = 0; j < B_col; ++j){
      // 64個の数字ごとに格納
      // column orderであることに留意

      uint64_t part[4] = {};

      for(size_t p = 0; p < 4; ++p){
        for(size_t k = p * 64; k < (p+1)*64; ++k){
          unsigned int sign = i+k >= B_row ? 0 : (*(B.data() + (i+k) + j * B_row) > 0);
          part[p] = (part[p] << 1) | sign;
        }
      }

      B_arr[ (i/256) + j * (B_row_inc/256) ] = _mm256_set_epi64x(part[0], part[1], part[2], part[3]);
    }
  }

  // tmp


  outC = MatrixXd(A_row, B_col);

  #pragma omp parallel for collapse(2)
  for(size_t i = 0; i < A_row; ++i){
    for(size_t j = 0; j < B_col; ++j){
      uint64_t h_dis_ij = 0;
      assert(A_col_inc == B_row_inc);
      for(size_t part = 0; part < ( A_col_inc / 256 );  ++part ){
        h_dis_ij += h_dis(A_arr[i * (A_col_inc/256) + part], B_arr[ j * (B_row_inc/256) + part ]);
      }

      int C_ = A_col;
      int h_dis_ij_ = h_dis_ij;

      outC.coeffRef(i,j) = C_ - 2 * h_dis_ij_;
    }
  }

  delete[] A_arr;
  delete[] B_arr;
}


void binary_mult_self(const Eigen::MatrixXd &A,
                      Eigen::MatrixXd &outB_Bt){

  auto A_row = A.rows();
  auto A_col = A.cols();

  // AのcolとBのrowを増やす必要がある
  auto A_col_inc = A_col % 256 == 0 ? A_col : (A_col - (A_col % 256)) + 256;

  assert(A_col_inc % 256 == 0);

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

  outB_Bt = MatrixXd(A_row, A_row);

#pragma omp parallel for collapse(2)
  for(size_t i = 0; i < A_row; ++i){
    for(size_t j = 0; j < A_row; ++j){
      uint64_t h_dis_ij = 0;
      for(size_t part = 0; part < ( A_col_inc / 256 );  ++part ){
        h_dis_ij += h_dis(A_arr[i * (A_col_inc/256) + part], A_arr[ j * (A_col_inc/256) + part ]);
      }

      int C_ = A_col;
      int h_dis_ij_ = h_dis_ij;

      outB_Bt.coeffRef(i,j) = C_ - 2 * h_dis_ij_;
    }
  }
}

__m512i* alloc_tmp(size_t size){
  static __m512i* tmp = nullptr;
  if(tmp == nullptr){
    tmp = new __m512i[ size ];
  }
  return tmp;
}

void binary_mult512_self(const Eigen::MatrixXd &A,
                 Eigen::MatrixXd &outC) {


  auto A_row = A.rows();
  auto A_col = A.cols();

  // AのcolとBのrowを増やす必要がある
  auto A_col_inc = A_col % 512 == 0 ? A_col : (A_col - (A_col % 512)) + 512;

  assert(A_col_inc % 512 == 0);

  auto A_arr = alloc_tmp(A_row * (A_col_inc / 512));

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
