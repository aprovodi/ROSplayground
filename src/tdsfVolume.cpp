#include "kfusionCPU/tdsfVolume.h"
#include "kfusionCPU/kfusionCPU.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume::TsdfVolume(const Eigen::Vector3i& resolution) :
    resolution_(resolution)
{
    const Eigen::Vector3f default_volume_size = Eigen::Vector3f::Constant(3.f); //meters
    const float default_pos_tranc_dist = 0.1f; //meters
    const float default_neg_tranc_dist = -0.03f; //meters

    setSize(default_volume_size);

    cell_size_ = size_.array() / resolution_.array().cast<float> ();

    setPositiveTsdfTruncDist(default_pos_tranc_dist);
    setNegativeTsdfTruncDist(default_neg_tranc_dist);

    num_cells_ = resolution_(0) * resolution_(1) * resolution_(2);

    tsdf_ = new TsdfCell[num_cells_];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume::~TsdfVolume()
{
     //delete[] tsdf_;
     tsdf_ = NULL;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::reset()
{
    std::fill(begin(), end(), TsdfCell(pos_tranc_dist_, 0));
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
    size_ = size;
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
    return size_;
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
float cvpr_tum::TsdfVolume::getTSDFValue(const Eigen::Vector3f& glocation)
{
    if (std::isnan(glocation(0) + glocation(1) + glocation(2)))
        return pos_tranc_dist_;

    double i, j, k;
    float x = modf(glocation(0) / cell_size_(0) + resolution_(0) / 2.f, &i);
    float y = modf(glocation(1) / cell_size_(1) + resolution_(1) / 2.f, &j);
    float z = modf(glocation(2) / cell_size_(2) + resolution_(2) / 2.f, &k);

    if (i >= resolution_(0) - 1 || j >= resolution_(1) - 1 || k >= resolution_(2) - 1 || i < 0 || j < 0 || k < 0)
        return pos_tranc_dist_;

    int I = int(i);
    int J = int(j);
    int K = int(k);

    float a1 = v(I, J, K) * (1 - z) + v(I, J, K + 1) * z;
    float a2 = v(I, J + 1, K) * (1 - z) + v(I, J + 1, K + 1) * z;
    float b1 = v(I + 1, J, K) * (1 - z) + v(I + 1, J, K + 1) * z;
    float b2 = v(I + 1, J + 1, K) * (1 - z) + v(I + 1, J + 1, K + 1) * z;

    return (a1 * (1 - y) + a2 * y) * (1 - x) + (b1 * (1 - y) + b2 * y) * x;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float cvpr_tum::TsdfVolume::getTSDFGradient(const Eigen::Vector3f& glocation, int stepSize, int dim)
{
    float delta = cell_size_(dim) * stepSize;
    Eigen::Vector3f location_offset = Eigen::Vector3f::Zero();
    location_offset(dim) = delta;

    return ((getTSDFValue(glocation + location_offset)) - (getTSDFValue(glocation - location_offset))) / (2.0 * delta);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::integrate(const cv::Mat& raw_depth_map, const Intr& intr, const Eigen::Matrix4f& camtoworld)
{
    std::cout << resolution_(0) << " " << resolution_(1) << " " << resolution_(2) << std::endl;
    for (int x = 0; x < resolution_(0); ++x)
    {
        for (int y = 0; y < resolution_(1); ++y)
        {
            for (int z = 0; z < resolution_(2); ++z)
            {
                Eigen::Vector3f v_g(getVoxelGCoo(x, y, z));

                //tranform to cam coo space
                Eigen::Vector4f tmp;
                tmp << v_g(0), v_g(1), v_g(2), 1.f;
                Eigen::Vector4f v_new = camtoworld * tmp;
                Eigen::Vector3f v(v_new(0), v_new(1), v_new(2));

                Eigen::Vector4f camera = camtoworld * Eigen::Vector4f(0.0, 0.0, 0.0, 1.0);

                if (v(2) - camera(2) < 0)
                    continue;

                cv::Point2d coo(0, 0); //project to current cam
                coo.x = v(0) * intr.fx / v(2) + intr.cx;
                coo.y = v(1) * intr.fy / v(2) + intr.cy;

                if (v(2) > 0 && coo.x >= 0 && coo.y >= 0 && coo.x < raw_depth_map.cols && coo.y < raw_depth_map.rows
                        && !std::isnan(v(2)))
                {
                    float Dp = raw_depth_map.at<float> (coo.y, coo.x);

                    const float W = 1 / ((1 + Dp) * (1 + Dp));

                    float Eta(Dp - v(2));

                    if (Eta >= neg_tranc_dist_)// && Eta<Dmax)
                    {

                        float D = fmin(Eta, pos_tranc_dist_);//*copysign(1.0,Eta);*perpendicular

                        Eigen::Vector3i pos(x, y, z);
                        TsdfCell data = operator[](pos);
                        float tsdf_prev = data.first;
                        float weight_prev = data.second;

                        float tsdf_new = (tsdf_prev * weight_prev + D * W) / (weight_prev + W);
                        float weight_new = fmin(weight_prev + W, MAX_WEIGHT);

                        set(x, y, z, TsdfCell(tsdf_new, weight_new));
                    }
                    /*
                     if (Dp != 0)
                     {
                     float xl = (coo.x - intr.cx) / intr.fx;
                     float yl = (coo.y - intr.cy) / intr.fy;
                     float lambda_inv = sqrtf(xl * xl + yl * yl + 1);

                     Eigen::Vector3f diff = tcam - v_g;
                     float sdf = diff.norm() * lambda_inv - Dp;

                     sdf *= (-1);

                     if (sdf >= neg_tranc_dist_)
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
                     */
                }//within camera view
            }//z
        }//y
    }//x
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
;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume::TsdfCell& cvpr_tum::TsdfVolume::operator[](const Eigen::Vector3i& p) const
{
    if (p(0) >= resolution_(0) - 1 || p(1) >= resolution_(1) - 1 || p(2) >= resolution_(2) - 1 || p(0) < 0 || p(1) < 0 || p(2) < 0)
        return tsdf_[0];
    else return tsdf_[p(0) + p(1) * resolution_(0) + p(2) * resolution_(1) * resolution_(2)];
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float cvpr_tum::TsdfVolume::v(int pos_i, int pos_j, int pos_k) const
{
    return tsdf_[pos_i + pos_j * resolution_(0) + pos_k * resolution_(1) * resolution_(2)].first;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::TsdfVolume::set(int px, int py, int pz, const TsdfCell& d)
{
    tsdf_[px + py * resolution_(0) + pz * resolution_(0) * resolution_(1)] = d;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3f cvpr_tum::TsdfVolume::getVoxelGCoo(int x, int y, int z) const
{
    Eigen::Vector3f coo((float)x, (float)y, (float)z);
    coo.array() -= resolution_.array().cast<float> () / 2.f; //shift to the center;
    coo.array() *= cell_size_.array();
    return coo;
}
