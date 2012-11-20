#include "kfusionCPU/tdsfVolume.h"
#include "kfusionCPU/kfusionCPU.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume::TsdfVolume(const Eigen::Vector3i& resolution) :
    resolution_(resolution)
{
    int volume_x = resolution_(0);
    int volume_y = resolution_(1);
    int volume_z = resolution_(2);

    const Eigen::Vector3f default_volume_size = Eigen::Vector3f::Constant(3.f); //meters
    const float default_pos_tranc_dist = 0.1f; //meters
    const float default_neg_tranc_dist = -0.03f; //meters

    setSize(default_volume_size);

    cell_size_(0) = size_(0) / resolution_(0);
    cell_size_(1) = size_(1) / resolution_(1);
    cell_size_(2) = size_(2) / resolution_(2);

    setPositiveTsdfTruncDist(default_pos_tranc_dist);
    setNegativeTsdfTruncDist(default_neg_tranc_dist);

    tsdf_.resize(boost::extents[volume_x][volume_y][volume_z]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::setSize(const Eigen::Vector3f& size)
{
    size_ = size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::setPositiveTsdfTruncDist(float distance)
{
    pos_tranc_dist_ = std::max(distance, 2.1f * std::max(cell_size_(0), std::max(cell_size_(1), cell_size_(2))));

    /*if (tranc_dist_ != distance)
     ("Tsdf truncation distance can't be less than 2 * voxel_size. Passed value '%f', but setting minimal possible '%f'.\n", distance, tranc_dist_) */
}

void cvpr_tum::TsdfVolume::setNegativeTsdfTruncDist(float distance)
{
    neg_tranc_dist_ = -std::max(-distance, 2.1f * std::max(cell_size_(0), std::max(cell_size_(1), cell_size_(2))));

    /*if (tranc_dist_ != distance)
     ("Tsdf truncation distance can't be less than 2 * voxel_size. Passed value '%f', but setting minimal possible '%f'.\n", distance, tranc_dist_) */
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const cvpr_tum::TsdfVolume::TsdfVolumeData& cvpr_tum::TsdfVolume::data() const
{
    return tsdf_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const Eigen::Vector3f& cvpr_tum::TsdfVolume::getSize() const
{
    return size_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const Eigen::Vector3i& cvpr_tum::TsdfVolume::getResolution() const
{
    return resolution_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const Eigen::Vector3f cvpr_tum::TsdfVolume::getVoxelSize() const
{
    return size_.array() / resolution_.array().cast<float> ();
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

Eigen::Vector3f cvpr_tum::TsdfVolume::getVoxelGCoo(int x, int y, int z) const
{
    Eigen::Vector3f coo(x, y, z);
    coo.array() += 0.5f; //shift to cell center;

    coo.cwiseProduct(cell_size_);
    return coo;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::integrate(const cv::Mat& raw_depth_map, const Intr& intr, const Eigen::Matrix<float, 3, 3,
        Eigen::RowMajor>& Rcam_inv, const Eigen::Vector3f& tcam)
{
    for (int x = 0; x < resolution_(0); ++x)
    {
        for (int y = 0; y < resolution_(1); ++y)
        {
            for (int z = 0; z < resolution_(2); ++z)
            {
                Eigen::Vector3f v_g(getVoxelGCoo(x, y, z));

                //tranform to cam coo space
                Eigen::Vector3f v = Rcam_inv * (v_g - tcam);

                cv::Point2d coo(0, 0); //project to current cam
                coo.x = v(0) * intr.fx / v(2) + intr.cx;
                coo.y = v(1) * intr.fy / v(2) + intr.cy;

                if (v(2) > 0 && coo.x >= 0 && coo.y >= 0 && coo.x < raw_depth_map.cols && coo.y < raw_depth_map.rows)
                {
                    float Dp = raw_depth_map.at<float> (coo.y, coo.x);

                    if (Dp != 0)
                    {
                        float xl = (coo.x - intr.cx) / intr.fx;
                        float yl = (coo.y - intr.cy) / intr.fy;
                        float lambda_inv = sqrtf(xl * xl + yl * yl + 1);

                        Eigen::Vector3f diff = tcam - v_g;
                        float sdf = diff.norm() * lambda_inv - Dp;

                        sdf *= (-1);

                        if (sdf >= -neg_tranc_dist_)
                        {
                            float tsdf = fmin(1, sdf / pos_tranc_dist_);

                            int weight_prev = tsdf_[x][y][z].first;
                            float tsdf_prev = tsdf_[x][y][z].second;

                            const int Wrk = 1;

                            float tsdf_new = (tsdf_prev * weight_prev + Wrk * tsdf) / (weight_prev + Wrk);
                            int weight_new = fmin(weight_prev + Wrk, MAX_WEIGHT);

                            tsdf_[x][y][z].first = tsdf_new;
                            tsdf_[x][y][z].second = weight_new;
                        }
                    }
                }//within camera view
            }//z
        }//y
    }//x
}
