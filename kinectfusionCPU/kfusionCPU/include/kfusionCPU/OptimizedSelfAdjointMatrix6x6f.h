/*
 * OptimizedSelfAdjointMatrix6x6f.h
 *
 *  Created on: Feb 13, 2013
 *      Author: provodin
 */

#ifndef OPTIMIZEDSELFADJOINTMATRIX6X6F_H_
#define OPTIMIZEDSELFADJOINTMATRIX6X6F_H_

#include <Eigen/Core>

namespace cvpr_tum
{

class OptimizedSelfAdjointMatrix6x6f
{
public:
    void rankUpdate(const Eigen::Matrix<float, 6, 1>& u, const float& alpha);

    void operator +=(const OptimizedSelfAdjointMatrix6x6f& other);

    void setZero();

    void toEigen(Eigen::Matrix<float, 6, 6>& m) const;

private:
    enum {
      Size = 24
    };
    EIGEN_ALIGN16 float data[Size];

};

} /* namespace cvpr_tum */
#endif /* OPTIMIZEDSELFADJOINTMATRIX6X6F_H_ */
