/*
 * OptimizedSelfAdjointMatrix6x6f.cpp
 *
 *  Created on: Feb 13, 2013
 *      Author: provodin
 */

#include "kfusionCPU/OptimizedSelfAdjointMatrix6x6f.h"

#include <immintrin.h>
#include <pmmintrin.h>

namespace cvpr_tum
{


void OptimizedSelfAdjointMatrix6x6f::setZero()
{
  for(size_t idx = 0; idx < Size; idx++)
    data[idx] = 0.0f;
}

void OptimizedSelfAdjointMatrix6x6f::rankUpdate(const Eigen::Matrix<float, 6, 1>& u, const float& alpha)
{
  __m128 s = _mm_set1_ps(alpha);
  __m128 v1234 = _mm_loadu_ps(u.data());
  __m128 v56xx = _mm_loadu_ps(u.data() + 4);

  __m128 v1212 = _mm_movelh_ps(v1234, v1234);
  __m128 v3434 = _mm_movehl_ps(v1234, v1234);
  __m128 v5656 = _mm_movelh_ps(v56xx, v56xx);

  __m128 v1122 = _mm_mul_ps(s, _mm_unpacklo_ps(v1212, v1212));

  _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), _mm_mul_ps(v1122, v1212)));
  _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), _mm_mul_ps(v1122, v3434)));
  _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), _mm_mul_ps(v1122, v5656)));

  __m128 v3344 = _mm_mul_ps(s, _mm_unpacklo_ps(v3434, v3434));

  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_mul_ps(v3344, v3434)));
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_mul_ps(v3344, v5656)));

  __m128 v5566 = _mm_mul_ps(s, _mm_unpacklo_ps(v5656, v5656));

  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_mul_ps(v5566, v5656)));
}

void OptimizedSelfAdjointMatrix6x6f::operator +=(const OptimizedSelfAdjointMatrix6x6f& other)
{
  _mm_store_ps(data + 0, _mm_add_ps(_mm_load_ps(data + 0), _mm_load_ps(other.data + 0)));
  _mm_store_ps(data + 4, _mm_add_ps(_mm_load_ps(data + 4), _mm_load_ps(other.data + 4)));
  _mm_store_ps(data + 8, _mm_add_ps(_mm_load_ps(data + 8), _mm_load_ps(other.data + 8)));
  _mm_store_ps(data + 12, _mm_add_ps(_mm_load_ps(data + 12), _mm_load_ps(other.data + 12)));
  _mm_store_ps(data + 16, _mm_add_ps(_mm_load_ps(data + 16), _mm_load_ps(other.data + 16)));
  _mm_store_ps(data + 20, _mm_add_ps(_mm_load_ps(data + 20), _mm_load_ps(other.data + 20)));
}

void OptimizedSelfAdjointMatrix6x6f::toEigen(Eigen::Matrix<float, 6, 6>& m) const
{
  Eigen::Matrix<float, 6, 6> tmp;
  size_t idx = 0;

  for(size_t i = 0; i < 6; i += 2)
  {
    for(size_t j = i; j < 6; j += 2)
    {
      tmp(i , j ) = data[idx++];
      tmp(i , j+1) = data[idx++];
      tmp(i+1, j ) = data[idx++];
      tmp(i+1, j+1) = data[idx++];
    }
  }

  tmp.selfadjointView<Eigen::Upper>().evalTo(m);
}

} /* namespace cvpr_tum */
