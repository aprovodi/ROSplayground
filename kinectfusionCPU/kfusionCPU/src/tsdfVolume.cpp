#include "kfusionCPU/tsdfVolume.h"
#include "kfusionCPU/CircularBuffer.h"

////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume::TsdfVolume(const Eigen::Vector3i& resolution)
    : resolution_(resolution), num_columns_(resolution(0)),
      num_rows_(resolution(1)), num_slices_(resolution(2)) {

    std::cout << "TSDF VOLUME CONSTRUCTOR WAS CALLED" << std::endl;

    // all values are in meters
    const Eigen::Vector3f default_volume_size = Eigen::Vector3f::Constant(3.f);
    const float default_pos_tranc_dist = 0.04f;
    const float default_neg_tranc_dist = -0.04f;

    setSize(default_volume_size);

    setPositiveTsdfTruncDist(default_pos_tranc_dist);
    setNegativeTsdfTruncDist(default_neg_tranc_dist);

    num_cells_ = num_columns_ * num_rows_ * num_slices_;

    tsdf_ = new float[num_cells_];
    weights_ = new float[num_cells_];

    reset();
}

////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume::~TsdfVolume() {
    std::cout << "TSDF VOLUME DESTRUCTOR WAS CALLED" << std::endl;
    delete[] tsdf_;
    tsdf_ = NULL;
    delete[] weights_;
    weights_ = NULL;
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::release() {
    delete[] tsdf_;
    tsdf_ = NULL;
    delete[] weights_;
    weights_ = NULL;
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::reset() {
    std::fill(tsdf_, tsdf_ + num_cells_, pos_tranc_dist_);
    std::fill(weights_, weights_ + num_cells_, 0.f);
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::copyFrom(float *ptr_arg, unsigned int data_size_arg) {
    if (data_size_arg <= num_cells_)
        std::copy(ptr_arg, ptr_arg + data_size_arg, tsdf_);
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::setSize(const Eigen::Vector3f& size) {
    world_size_ = size;
    cell_size_ = world_size_.array() / resolution_.array().cast<float> ();
    inverse_cell_size_ = resolution_.array().cast<float> () / world_size_.array();
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::setPositiveTsdfTruncDist(float distance) {
    pos_tranc_dist_ = std::max(distance, 2.1f * std::max(cell_size_(0), std::max(cell_size_(1), cell_size_(2))));

    /*if (pos_tranc_dist_ != distance)
     ("Tsdf truncation distance can't be less than 2 * voxel_size. Passed value '%f', but setting minimal possible '%f'.\n", distance, tranc_dist_) */
}

void cvpr_tum::TsdfVolume::setNegativeTsdfTruncDist(float distance) {
    neg_tranc_dist_ = -std::max(-distance, 2.1f * std::max(cell_size_(0), std::max(cell_size_(1), cell_size_(2))));

    /*if (tranc_dist_ != distance)
     ("Tsdf truncation distance can't be less than 2 * voxel_size. Passed value '%f', but setting minimal possible '%f'.\n", distance, tranc_dist_) */
}

////////////////////////////////////////////////////////////////////////////////
float cvpr_tum::TsdfVolume::getInterpolatedTSDFValue(const Eigen::Vector3f& glocation) {
    if (std::isnan(glocation(0) + glocation(1) + glocation(2)))
        return pos_tranc_dist_;

    float glocation_scaled_x = glocation(0) * inverse_cell_size_(0) + (resolution_(0) >> 1);
    float glocation_scaled_y = glocation(1) * inverse_cell_size_(1) + (resolution_(1) >> 1);
    float glocation_scaled_z = glocation(2) * inverse_cell_size_(2) + (resolution_(2) >> 1);

    unsigned int I = floorf(glocation_scaled_x);        // round to negative infinity // or we could use int()?
    unsigned int J = floorf(glocation_scaled_y);
    unsigned int K = floorf(glocation_scaled_z);

    if (I >= num_columns_ - 1 || J >= num_rows_ - 1 || K >= num_slices_ - 1)
        return pos_tranc_dist_;

    float xd = glocation_scaled_x - I;
    float yd = glocation_scaled_y - J;
    float zd = glocation_scaled_z - K;

    unsigned int Z1 = num_columns_ * ( num_rows_ * (K + 0) + (J + 0) ) + I;
    unsigned int Z2 = num_columns_ * ( num_rows_ * (K + 0) + (J + 1) ) + I;
    unsigned int Z3 = num_columns_ * ( num_rows_ * (K + 1) + (J + 0) ) + I;
    unsigned int Z4 = num_columns_ * ( num_rows_ * (K + 1) + (J + 1) ) + I;

    float c00 = tsdf_[Z1] * (1 - xd) + tsdf_[Z1 + 1] * xd;
    float c10 = tsdf_[Z2] * (1 - xd) + tsdf_[Z2 + 1] * xd;
    float c01 = tsdf_[Z3] * (1 - xd) + tsdf_[Z3 + 1] * xd;
    float c11 = tsdf_[Z4] * (1 - xd) + tsdf_[Z4 + 1] * xd;

    float c0 = c00 * (1 - yd) + c10 * yd;
    float c1 = c01 * (1 - yd) + c11 * yd;

    float c = c0 * (1 - zd) + c1 * zd;
    return c;
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<float, 1, 3> cvpr_tum::TsdfVolume::getTSDFGradient(const Eigen::Vector3f& glocation) {
    Eigen::Vector3f tmp = Eigen::Vector3f::Zero();
    int stepSize = 1;
    Eigen::Matrix<float, 1, 3> gradientVector;

    float delta = cell_size_(0) * stepSize;
    tmp = glocation;
    tmp(0) += delta;
    float Fx1 = getInterpolatedTSDFValue(tmp);

    tmp = glocation;
    tmp(0) -= delta;
    float Fx2 = getInterpolatedTSDFValue(tmp);

    gradientVector(0) = (Fx1 - Fx2) / (2.f * delta);

    delta = cell_size_(1) * stepSize;
    tmp = glocation;
    tmp(1) += delta;
    float Fy1 = getInterpolatedTSDFValue(tmp);

    tmp = glocation;
    tmp(1) -= delta;
    float Fy2 = getInterpolatedTSDFValue(tmp);

    gradientVector(1) = (Fy1 - Fy2) / (2.f * delta);

    delta = cell_size_(2) * stepSize;
    tmp = glocation;
    tmp(2) += delta;
    float Fz1 = getInterpolatedTSDFValue(tmp);

    tmp = glocation;
    tmp(2) -= delta;
    float Fz2 = getInterpolatedTSDFValue(tmp);

    gradientVector(2) = (Fz1 - Fz2) / (2.f * delta);

    return gradientVector;
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::integrate(const cv::Mat& raw_depth_map,
                                     const Intr& intr,
                                     const Eigen::Matrix3f& Rcam_inv,
                                     const Eigen::Vector3f& tcam, CircularBuffer* buffer) {
    unsigned int idx_curr = 0;
    for (unsigned int x = 0; x < num_columns_; x++) {
        for (unsigned int y = 0; y < num_rows_; y++) {
            for (unsigned int z = 0; z < num_slices_; ++z) {
                idx_curr = num_columns_ * (z * num_rows_ + y) + x;

                // opt: precompute step and increment as you do with x, y, z
                Eigen::Vector3f v_g(getVoxelGCoo(x, y, z));

                // tranform to cam coo space
                Eigen::Vector3f v = Rcam_inv * (v_g - tcam);

                if (v(2) - tcam(2) < 0)
                    continue;

                float* pos_value = tsdf_ + idx_curr;
                float* pos_weight = weights_ + idx_curr;
                // shift the pointer to relative indices
                //buffer->shift_tsdf_pointer(&pos_value, &pos_weight);

                // As the pointer is incremented in the for loop, we have to make sure that the pointer is never outside the memory
                if(pos_value > end())
                {
                    pos_value -= (end() - begin()); pos_weight -= (end_weights() - begin_weights());
                }

                // project to current cam
                // opt: pre-multiply intrinsic and transformation matrix
                int coo_x = floorf(v(0) * intr.fx / v(2) + intr.cx);
                int coo_y = floorf(v(1) * intr.fy / v(2) + intr.cy);

                // within camera view
                if (coo_x < 0 || coo_y < 0 || coo_x > raw_depth_map.cols - 1 || coo_y > raw_depth_map.rows - 1)
                    continue;

                float Dp = raw_depth_map.at<float>(coo_y, coo_x);

                if (Dp != 0) {          // check validity here
                    // opt: from here on its easy to SSE
                    float Eta(Dp - v(2));

                    if (Eta >= neg_tranc_dist_) {
                        float D = fmin(Eta, pos_tranc_dist_);

                        float tsdf_prev = *pos_value;
                        float weight_prev = *pos_weight;

                        const float W = 1 / ((1 + Dp) * (1 + Dp));

                        float tsdf_new = (tsdf_prev * weight_prev + D * W) / (weight_prev + W);
                        float weight_new = fmin(weight_prev + W, MAX_WEIGHT);

                        *pos_value = tsdf_new;
                        *pos_weight = weight_new;
                    }
                }
            } // z
        } // y
    }// x
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3f cvpr_tum::TsdfVolume::getVoxelGCoo(int x, int y, int z) const {
    //    Eigen::Vector3f coo((float)x, (float)y, (float)z);
    float ray_x = (x - resolution_(0) / 2) * cell_size_(0);
    float ray_y = (y - resolution_(1) / 2) * cell_size_(1);
    float ray_z = (z - resolution_(2) / 2) * cell_size_(2);
    //    coo.array() -= resolution_.array().cast<float> () / 2.f; //shift to the center;
    //    coo.array() *= cell_size_.array();
    //    return coo;
    return Eigen::Vector3f(ray_x, ray_y, ray_z);
}

////////////////////////////////////////////////////////////////////////////////
float cvpr_tum::TsdfVolume::v(int pos_x, int pos_y, int pos_z) const
{
    return tsdf_[num_columns_ * (num_rows_ * pos_z + pos_y) + pos_x];
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::setv(int pos_x, int pos_y, int pos_z, float val)
{
    int pos = num_columns_ * (num_rows_ * pos_z + pos_y) + pos_x;
    tsdf_[pos] = val;
    weights_[pos] = 0.f;
}

////////////////////////////////////////////////////////////////////////////////
bool cvpr_tum::TsdfVolume::validGradient(const Eigen::Vector3f& glocation) {
    /*
     The function tests the current location and its adjacent
     voxels for valid values (written at least once) to
     determine if derivatives at this location are
     computable in all three directions.

     Since the function SDF(Eigen::Vector3f& glocation) is a
     trilinear interpolation between neighbours, testing the
     validity of the gradient involves looking at all the
     values that would contribute to the final  gradient.
     If any of these have a weight equal to zero, the result
     is false.
                   X--------X
                 /        / |
               X--------X   ----X
               |        |   | / |
           X----        |   X-------X
         /     |        | /       / |
       X-------X--------X-------X   |
       |     /        / |       |   |
       |   X--------X   |       |   |
  J    |   |        |   |       | /
  ^    X----        |   X-------X
  |        |        | / |  |
   --->I   X--------X   |  X
 /             |        | /
v              X--------X
K                                                */
    if (std::isnan(glocation(0) + glocation(1) + glocation(2)))
        return false;

    float glocation_scaled_x = glocation(0) / cell_size_(0);// + resolution_(0) / 2.f;
    float glocation_scaled_y = glocation(1) / cell_size_(1);// + resolution_(1) / 2.f;
    float glocation_scaled_z = glocation(2) / cell_size_(2);// + resolution_(2) / 2.f;

    int I = floorf(glocation_scaled_x);
    int J = floorf(glocation_scaled_y);
    int K = floorf(glocation_scaled_z);

    if (I >= resolution_(0) - 3 || J >= resolution_(1) - 3 || K >= resolution_(2) - 3 || I <= 1 || J <= 1 || K <= 1)
        return false;

    unsigned int D10 = (K + 1) * num_rows_ * num_slices_ + (J + 0) * num_slices_ + I;
    unsigned int D20 = (K + 2) * num_rows_ * num_slices_ + (J + 0) * num_slices_ + I;

    unsigned int D01 = (K + 0) * num_rows_ * num_slices_ + (J + 1) * num_slices_ + I;
    unsigned int D11 = (K + 1) * num_rows_ * num_slices_ + (J + 1) * num_slices_ + I;
    unsigned int D21 = (K + 2) * num_rows_ * num_slices_ + (J + 1) * num_slices_ + I;
    unsigned int D31 = (K + 3) * num_rows_ * num_slices_ + (J + 1) * num_slices_ + I;

    unsigned int D02 = (K + 0) * num_rows_ * num_slices_ + (J + 2) * num_slices_ + I;
    unsigned int D12 = (K + 1) * num_rows_ * num_slices_ + (J + 2) * num_slices_ + I;
    unsigned int D22 = (K + 2) * num_rows_ * num_slices_ + (J + 2) * num_slices_ + I;
    unsigned int D32 = (K + 3) * num_rows_ * num_slices_ + (J + 2) * num_slices_ + I;

    unsigned int D13 = (K + 1) * num_rows_ * num_slices_ + (J + 3) * num_slices_ + I;
    unsigned int D23 = (K + 2) * num_rows_ * num_slices_ + (J + 3) * num_slices_ + I;

    if (                              !tsdf_[D10 + 1] == pos_tranc_dist_ || !tsdf_[D10 + 2] == pos_tranc_dist_ ||
                                      !tsdf_[D20 + 1] == pos_tranc_dist_ || !tsdf_[D20 + 2] == pos_tranc_dist_ ||

                                      !tsdf_[D01 + 1] == pos_tranc_dist_ || !tsdf_[D01 + 2] == pos_tranc_dist_ ||
                                      !tsdf_[D11] == pos_tranc_dist_ || !tsdf_[D11 + 1] == pos_tranc_dist_ || !tsdf_[D11 + 2] == pos_tranc_dist_ || !tsdf_[D11 + 3] == pos_tranc_dist_ ||
                                      !tsdf_[D21] == pos_tranc_dist_ || !tsdf_[D21 + 1] == pos_tranc_dist_ || !tsdf_[D21 + 2] == pos_tranc_dist_ || !tsdf_[D21 + 3] == pos_tranc_dist_ ||
                                      !tsdf_[D31 + 1] == pos_tranc_dist_ || !tsdf_[D31 + 2] == pos_tranc_dist_ ||

                                      !tsdf_[D02 + 1] == pos_tranc_dist_ || !tsdf_[D02 + 2] == pos_tranc_dist_ ||
                                      !tsdf_[D12] == pos_tranc_dist_ || !tsdf_[D12 + 1] == pos_tranc_dist_ || !tsdf_[D12 + 2] == pos_tranc_dist_ || !tsdf_[D12 + 3] == pos_tranc_dist_ ||
                                      !tsdf_[D22] == pos_tranc_dist_ || !tsdf_[D22 + 1] == pos_tranc_dist_ || !tsdf_[D22 + 2] == pos_tranc_dist_ || !tsdf_[D22 + 3] == pos_tranc_dist_ ||
                                      !tsdf_[D32 + 1] == pos_tranc_dist_ || !tsdf_[D32 + 2] == pos_tranc_dist_ ||

                                      !tsdf_[D13 + 1] == pos_tranc_dist_ || !tsdf_[D13 + 2] == pos_tranc_dist_ ||
                                      !tsdf_[D23 + 1] == pos_tranc_dist_ || !tsdf_[D23 + 2] == pos_tranc_dist_
                                      ) return false;
    else
        return true;
}
