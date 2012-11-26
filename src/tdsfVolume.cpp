#include "kfusionCPU/tsdfVolume.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume::TsdfVolume(const Eigen::Vector3i& resolution) :
    resolution_(resolution), num_columns_(resolution(0)), num_rows_(resolution(1)), num_slices_(resolution(2))
{
    std::cout << "TSDF VOLUME CONSTRUCTOR WAS CALLED" << std::endl;

    const Eigen::Vector3f default_volume_size = Eigen::Vector3f::Constant(3.f); //meters
    const float default_pos_tranc_dist = 0.1f; //meters
    const float default_neg_tranc_dist = -0.04f; //meters

    setSize(default_volume_size);

    setPositiveTsdfTruncDist(default_pos_tranc_dist);
    setNegativeTsdfTruncDist(default_neg_tranc_dist);

    num_cells_ = num_columns_ * num_rows_ * num_slices_;

    tsdf_ = new float[num_cells_];
    weights_ = new float[num_cells_];
    reset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume::~TsdfVolume()
{
    std::cout << "TSDF VOLUME DESTRUCTOR WAS CALLED" << std::endl;
}

void cvpr_tum::TsdfVolume::release()
{
    delete[] tsdf_;
    tsdf_ = NULL;
    delete[] weights_;
    weights_ = NULL;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::reset()
{
    std::fill(tsdf_, tsdf_ + num_cells_, pos_tranc_dist_);
    std::fill(weights_, weights_ + num_cells_, 0.f);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume::iterator cvpr_tum::TsdfVolume::begin()
{
    return tsdf_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume::iterator cvpr_tum::TsdfVolume::end()
{
    return tsdf_ + num_cells_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::setSize(const Eigen::Vector3f& size)
{
    world_size_ = size;
    cell_size_ = world_size_.array() / resolution_.array().cast<float> ();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::setPositiveTsdfTruncDist(float distance)
{
    pos_tranc_dist_ = std::max(distance, 2.1f * std::max(cell_size_(0), std::max(cell_size_(1), cell_size_(2))));

    /*if (pos_tranc_dist_ != distance)
     ("Tsdf truncation distance can't be less than 2 * voxel_size. Passed value '%f', but setting minimal possible '%f'.\n", distance, tranc_dist_) */
}

void cvpr_tum::TsdfVolume::setNegativeTsdfTruncDist(float distance)
{
    neg_tranc_dist_ = -std::max(-distance, 2.1f * std::max(cell_size_(0), std::max(cell_size_(1), cell_size_(2))));

    /*if (tranc_dist_ != distance)
     ("Tsdf truncation distance can't be less than 2 * voxel_size. Passed value '%f', but setting minimal possible '%f'.\n", distance, tranc_dist_) */
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const Eigen::Vector3f& cvpr_tum::TsdfVolume::getSize() const
{
    return world_size_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const Eigen::Vector3i& cvpr_tum::TsdfVolume::getResolution() const
{
    return resolution_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float cvpr_tum::TsdfVolume::getPositiveTsdfTruncDist() const
{
    return pos_tranc_dist_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float cvpr_tum::TsdfVolume::getNegativeTsdfTruncDist() const
{
    return neg_tranc_dist_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const Eigen::Vector3f& cvpr_tum::TsdfVolume::getVoxelSize() const
{
    return cell_size_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float cvpr_tum::TsdfVolume::getInterpolatedTSDFValue(const Eigen::Vector3f& glocation)
{
    if (std::isnan(glocation(0) + glocation(1) + glocation(2)))
        return pos_tranc_dist_;

    float glocation_scaled_x = glocation(0) / cell_size_(0) + resolution_(0) / 2.f;
    float glocation_scaled_y = glocation(1) / cell_size_(1) + resolution_(1) / 2.f;
    float glocation_scaled_z = glocation(2) / cell_size_(2) + resolution_(2) / 2.f;

    unsigned int I = floorf(glocation_scaled_x); // round to negative infinity // or we could use int()?
    unsigned int J = floorf(glocation_scaled_y);
    unsigned int K = floorf(glocation_scaled_z);

    if (I >= num_columns_ - 1 || J >= num_rows_ - 1 || K >= num_slices_ - 1)
        return pos_tranc_dist_;

    float xd = glocation_scaled_x - I;
    float yd = glocation_scaled_y - J;
    float zd = glocation_scaled_z - K;

    unsigned int N1 = (I + 0) * num_rows_ * num_slices_ + (J + 0) * num_slices_ + K;
    unsigned int N2 = (I + 0) * num_rows_ * num_slices_ + (J + 1) * num_slices_ + K;
    unsigned int N3 = (I + 1) * num_rows_ * num_slices_ + (J + 0) * num_slices_ + K;
    unsigned int N4 = (I + 1) * num_rows_ * num_slices_ + (J + 1) * num_slices_ + K;

    float c00 = tsdf_[N1] * (1 - zd) + tsdf_[N1 + 1] * zd;
    float c10 = tsdf_[N2] * (1 - zd) + tsdf_[N2 + 1] * zd;
    float c01 = tsdf_[N3] * (1 - zd) + tsdf_[N3 + 1] * zd;
    float c11 = tsdf_[N4] * (1 - zd) + tsdf_[N4 + 1] * zd;

    float c0 = c00 * (1 - yd) + c10 * yd;
    float c1 = c01 * (1 - yd) + c11 * yd;

    float c = c0 * (1 - xd) + c1 * xd;
    return c;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<float, 1, 3> cvpr_tum::TsdfVolume::getTSDFGradient(const Eigen::Vector3f& glocation)
{
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::integrate(const cv::Mat& raw_depth_map, const Intr& intr, const Eigen::Matrix3f& Rcam_inv, const Eigen::Vector3f& tcam)
{
    for (unsigned int x = 0; x < num_columns_; ++x)
    {
        for (unsigned int y = 0; y < num_rows_; ++y)
        {
            for (unsigned int z = 0; z < num_slices_; ++z)
            {
                unsigned int idx_curr = x * num_rows_ * num_slices_ + y * num_slices_ + z;

                Eigen::Vector3f v_g(getVoxelGCoo(x, y, z));

                //tranform to cam coo space
                Eigen::Vector3f v = Rcam_inv * (v_g - tcam);

                if (v(2) - tcam(2) < 0)
                    continue;

                //project to current cam
                int coo_x = floorf(v(0) * intr.fx / v(2) + intr.cx);
                int coo_y = floorf(v(1) * intr.fy / v(2) + intr.cy);

                if (coo_x > 0 && coo_y > 0 && coo_x < raw_depth_map.cols - 1 && coo_y < raw_depth_map.rows - 1)
                {
                    float Dp = raw_depth_map.at<float> (coo_y, coo_x);

                    if (Dp != 0)
                    {
                        const float W = 1 / ((1 + Dp) * (1 + Dp));

                        float Eta(Dp - v(2));

                        if (Eta >= neg_tranc_dist_)// && Eta < pos_tranc_dist_)
                        {

                            float D = fmin(Eta, pos_tranc_dist_);//*copysign(1.0,Eta);*perpendicular

                            float tsdf_prev = tsdf_[idx_curr];
                            float weight_prev = weights_[idx_curr];

                            float tsdf_new = (tsdf_prev * weight_prev + D * W) / (weight_prev + W);
                            float weight_new = fmin(weight_prev + W, MAX_WEIGHT);

                            tsdf_[idx_curr] = tsdf_new;
                            weights_[idx_curr] = weight_new;
                        }
                    }
                }//within camera view
            }//z
        }//y
    }//x
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float cvpr_tum::TsdfVolume::v(int pos_x, int pos_y, int pos_z) const
{
    return tsdf_[pos_z * num_rows_ * num_slices_ + pos_y * num_columns_ + pos_x];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool cvpr_tum::TsdfVolume::validGradient(const Eigen::Vector3f& glocation)
{
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

    double i, j, k;
    modf(glocation(0) / cell_size_(0), &i);
    modf(glocation(1) / cell_size_(1), &j);
    modf(glocation(2) / cell_size_(2), &k);

    if (std::isnan(i) || std::isnan(j) || std::isnan(k))
        return false;

    int I = int(i) - 1;
    int J = int(j) - 1;
    int K = int(k) - 1;

    if (I >= resolution_(0) - 4 || J >= resolution_(1) - 3 || K >= resolution_(2) - 3 || I <= 1 || J <= 1 || K <= 1)
        return false;

    if (!v(I + 1, J + 0, K + 1) == pos_tranc_dist_ || !v(I + 1, J + 0, K + 2) == pos_tranc_dist_ || !v(I + 2, J + 0, K
            + 1) == pos_tranc_dist_ || !v(I + 2, J + 0, K + 2) == pos_tranc_dist_ ||

    !v(I + 0, J + 1, K + 1) == pos_tranc_dist_ || !v(I + 0, J + 1, K + 2) == pos_tranc_dist_ || !v(I + 1, J + 1, K)
            == pos_tranc_dist_ || !v(I + 1, J + 1, K + 1) == pos_tranc_dist_ || !v(I + 1, J + 1, K + 2)
            == pos_tranc_dist_ || !v(I + 1, J + 1, K + 3) == pos_tranc_dist_ || !v(I + 2, J + 1, K) == pos_tranc_dist_
            || !v(I + 2, J + 1, K + 1) == pos_tranc_dist_ || !v(I + 2, J + 1, K + 2) == pos_tranc_dist_ || !v(I + 2, J
            + 1, K + 3) == pos_tranc_dist_ || !v(I + 3, J + 1, K + 1) == pos_tranc_dist_ || !v(I + 3, J + 1, K + 2)
            == pos_tranc_dist_ ||

    !v(I + 0, J + 2, K + 1) == pos_tranc_dist_ || !v(I + 0, J + 2, K + 2) == pos_tranc_dist_ || !v(I + 1, J + 2, K)
            == pos_tranc_dist_ || !v(I + 1, J + 2, K + 1) == pos_tranc_dist_ || !v(I + 1, J + 2, K + 2)
            == pos_tranc_dist_ || !v(I + 1, J + 2, K + 3) == pos_tranc_dist_ || !v(I + 2, J + 2, K) == pos_tranc_dist_
            || !v(I + 2, J + 2, K + 1) == pos_tranc_dist_ || !v(I + 2, J + 2, K + 2) == pos_tranc_dist_ || !v(I + 2, J
            + 2, K + 3) == pos_tranc_dist_ || !v(I + 3, J + 2, K + 1) == pos_tranc_dist_ || !v(I + 3, J + 2, K + 2)
            == pos_tranc_dist_ ||

    !v(I + 1, J + 3, K + 1) == pos_tranc_dist_ || !v(I + 1, J + 3, K + 2) == pos_tranc_dist_ || !v(I + 2, J + 3, K + 1)
            == pos_tranc_dist_ || !v(I + 2, J + 3, K + 2) == pos_tranc_dist_)
        return false;
    else
        return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3f cvpr_tum::TsdfVolume::getVoxelGCoo(int x, int y, int z) const
{
    //Eigen::Vector3f coo((float)x, (float)y, (float)z);
    float ray_x = (x - resolution_(0) / 2) * cell_size_(0);
    float ray_y = (y - resolution_(1) / 2) * cell_size_(1);
    float ray_z = (z - resolution_(2) / 2) * cell_size_(2);
    //coo.array() -= resolution_.array().cast<float> () / 2.f; //shift to the center;
    //coo.array() *= cell_size_.array();
    //return coo;
    return Eigen::Vector3f(ray_x, ray_y, ray_z);
}
