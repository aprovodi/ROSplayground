#include "kfusionCPU/kfusionCPU.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>

namespace cvpr_tum
{
template<typename T>
    bool is_nan(const T &value)
    {
        // True if NAN
        return value != value;
    }

Eigen::Matrix4f Twist(const kfusionCPU::Vector6f& xi)
{
    Eigen::Matrix4f M;

    M << 0.0, -xi(2), xi(1), xi(3), xi(2), 0.0, -xi(0), xi(4), -xi(1), xi(0), 0.0, xi(5), 0.0, 0.0, 0.0, 0.0;

    return M;
}
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::kfusionCPU::kfusionCPU(int rows, int cols) :
    rows_(rows), cols_(cols), global_time_(0), integration_metric_threshold_(0.f), robust_statistic_coefficient_(0.02f)
{
    const Vector3f volume_size = Vector3f::Constant(VOLUME_SIZE);
    const Eigen::Vector3i volume_resolution(VOLUME_X, VOLUME_Y, VOLUME_Z);

    tsdf_volume_ = TsdfVolume::Ptr(new TsdfVolume(volume_resolution));
    tsdf_volume_->setSize(volume_size);

    setDepthIntrinsics(520.f, 520.f, 319.5, 239.5); // default values, can be overwritten

    init_Rcam_ = Eigen::Matrix3f::Identity();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
    init_tcam_ = Eigen::Vector3f::Zero();//volume_size * 0.5f - Vector3f(0, 0, volume_size(2) / 2 * 1.2f);

    const int iters[] = {10, 5, 4};
    std::copy(iters, iters + LEVELS, icp_iterations_);

    allocateMaps(rows_, cols_);

    //rmats_.reserve(30000);
    //tvecs_.reserve(30000);

    first_frame_ = true;
    Pose_ << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    Transformation_ = Eigen::MatrixXf::Identity(4, 4);

    reset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::allocateMaps(int rows, int cols)
{
    for (int i = 0; i < LEVELS; ++i)
    {
        int pyr_rows = rows >> i;
        int pyr_cols = cols >> i;

        depth_pyramid_[i].create(pyr_rows, pyr_cols, raw_depth_map_encoding_);
        boost::array<int, 2> shape = { {pyr_rows, pyr_cols}};
        vertex_pyramid_[i].resize(shape);//boost::extents[pyr_rows][pyr_cols]);
        normal_pyramid_[i].resize(shape);//resize(boost::extents[pyr_rows][pyr_cols]);
    }

}

void cvpr_tum::kfusionCPU::reset()
{
    global_time_ = 0;
    rmats_.clear();
    tvecs_.clear();

    rmats_.push_back(init_Rcam_);
    tvecs_.push_back(init_tcam_);

    tsdf_volume_->reset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::setDepthIntrinsics(float fx, float fy, float cx, float cy)
{
    fx_ = fx;
    fy_ = fy;
    cx_ = (cx == -1) ? cols_ / 2 - 0.5f : cx;
    cy_ = (cy == -1) ? rows_ / 2 - 0.5f : cy;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::setInitalCameraPose(const Eigen::Affine3f& pose)
{
    init_Rcam_ = pose.rotation();
    init_tcam_ = pose.translation();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool cvpr_tum::kfusionCPU::operator()(const DepthMap& raw_depth)
{
    Intr intr(fx_, fy_, cx_, cy_);

    raw_depth_map_ = raw_depth;
    raw_depth_map_encoding_ = raw_depth.type();

    {
        ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");

        // 2. Apply bilinear filtering (using OpenCV)
        //cv::bilateralFilter(raw_depth, depth_pyramid_[0], D, sigma_colour, sigma_space);
        depth_pyramid_[0] = raw_depth;

        // 3. Compute depth pyramid.
        for (size_t i = 1; i < LEVELS; i++)
            cv::pyrDown(depth_pyramid_[i - 1], depth_pyramid_[i]);

        // 4. Compute vertex & normal pyramid.
        for (size_t i = 0; i < LEVELS; i++)
        {
            // 4.1 Populate the vertex map for the current level.
            create_vertex_map(intr(i), depth_pyramid_[i], vertex_pyramid_[i]);

            // 4.2 Populate the normal map for the current level.
            create_normal_map(vertex_pyramid_[i], normal_pyramid_[i]);
        }
    }

    // 5. ICP
    bool hasfused;
    if (!first_frame_)
    {
        ScopeTime time("Camera Tracking ...");
        hasfused = true;

        Vector6f xi;
        xi << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0; // + (Pose-previousPose)*0.1;
        Vector6f xi_prev = xi;

        std::cout << "Transformation" << Transformation_ << std::endl;

        for (int level_index = LEVELS - 1; level_index >= 0; --level_index)
        {
            int iter_num = icp_iterations_[level_index];

            VertexMap vmap_curr = vertex_pyramid_[level_index];

            for (int iter = 0; iter < iter_num; ++iter)
            {
                Eigen::Matrix4f camToWorld = Twist(xi).exp() * Transformation_;

                Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
                Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();

                ///////////////////////////
                for (size_t i = 0; i < vmap_curr.shape()[0]; i++)
                {
                    for (size_t j = 0; j < vmap_curr[0].shape()[1]; j++)
                    {
                        //tranform to global coo space
                        if (std::isnan(vmap_curr[i][j](2)) || vmap_curr[i][j](0) < 0.4)
                            continue;
                        Eigen::Vector4f tmp;
                        tmp << vmap_curr[i][j](0), vmap_curr[i][j](1), vmap_curr[i][j](2), 1.f;
                        Eigen::Vector4f v_g_new = camToWorld * tmp;
                        Eigen::Vector3f v_g(v_g_new(0), v_g_new(1), v_g_new(2));

                        //if(!tsdf_volume_->validGradient(v_g)) continue;

                        float D = tsdf_volume_->getTSDFValue(v_g);
                        if (fabs(D - tsdf_volume_->getPositiveTsdfTruncDist()) < std::numeric_limits<double>::epsilon()
                                || fabs(D - tsdf_volume_->getNegativeTsdfTruncDist())
                                        < std::numeric_limits<double>::epsilon())
                            continue;

                        //partial derivative of SDF wrt position
                        Eigen::Matrix<float, 1, 3> dSDF_dx(tsdf_volume_->getTSDFGradient(v_g, 1, 0),
                                                           tsdf_volume_->getTSDFGradient(v_g, 1, 1),
                                                           tsdf_volume_->getTSDFGradient(v_g, 1, 2));
                        //partial derivative of position wrt optimizaiton parameters
                        Eigen::Matrix<float, 3, 6> dx_dxi;
                        dx_dxi << 0, v_g(2), -v_g(1), 1, 0, 0, -v_g(2), 0, v_g(0), 0, 1, 0, v_g(1), -v_g(0), 0, 0, 0, 1;

                        //jacobian = derivative of SDF wrt xi (chain rule)
                        Eigen::Matrix<float, 1, 6> J = dSDF_dx * dx_dxi;

                        const float c = robust_statistic_coefficient_ * tsdf_volume_->getPositiveTsdfTruncDist();
                        float huber = fabs(D) < c ? 1.0 : c / fabs(D);

                        //Gauss - Newton approximation to hessian
                        A += huber * J.transpose() * J;
                        b += huber * J.transpose() * D;

                    }
                }

                ///////////////////////////

                double scaling = 1 / A.maxCoeff();

                b *= scaling;
                A *= scaling;
                const float regularization_ = 0.01;

                A = A + (regularization_) * Eigen::MatrixXf::Identity(6, 6);
                xi = xi - A.ldlt().solve(b);
                Vector6f Change = xi - xi_prev;
                double Cnorm = Change.norm();
                xi_prev = xi;
                if (Cnorm < 0.0001)
                    break;
            }
        }

        if (std::isnan(xi.sum()))
            xi << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        Pose_ = xi;
    }
    else
    {
        hasfused = false;
        first_frame_ = false;
    }

    Transformation_ = Twist(Pose_).exp() * Transformation_;

    transformations_.push_back(Transformation_);

    cumulative_pose_ += Pose_;
    Pose_ = Pose_ * 0.0;

    if (cumulative_pose_.norm() < 0.01 && hasfused)
    {
        return false;
    }
    cumulative_pose_ *= 0.0;

    {
        ScopeTime time(">>> Volume Integration");
        Eigen::Matrix4f camToWorld = Transformation_.inverse();

        tsdf_volume_->integrate(raw_depth_map_, intr, camToWorld);
    }

    return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::create_vertex_map(const Intr& intr, const DepthMap& src, VertexMap& dest)
{
    // get intrinsic parameters
    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    // for each pixel in the old depth map...
    for (size_t y = 0; y < height(src); y++)
    {
        for (size_t x = 0; x < width(src); x++)
        {
            // get the depth reading at this point
            float z = src.at<float> (y, x); // convert mm --> m

            // if we have some depth reading...
            //if (z != 0)
            //{
            dest[y][x](0) = z * (x - cx) / fx;
            dest[y][x](1) = z * (y - cy) / fy;
            dest[y][x](2) = z;
            /*
             }
             else
             {
             //std::cout << "we got something NAN at" << x << " " << y << std::endl;
             dest[y][x](0) = std::numeric_limits<float>::quiet_NaN();
             dest[y][x](1) = std::numeric_limits<float>::quiet_NaN();
             dest[y][x](2) = std::numeric_limits<float>::quiet_NaN();
             }
             */
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::create_normal_map(const VertexMap& src, NormalMap& dest)
{
    // for each vertex in the vertex map...
    for (size_t y = 0; y < height(src); y++)
    {
        for (size_t x = 0; x < width(src); x++)
        {
            if (x == width(src) - 1 || y == height(src) - 1)
            {
                dest[y][x](0) = std::numeric_limits<float>::quiet_NaN();
                dest[y][x](1) = std::numeric_limits<float>::quiet_NaN();
                dest[y][x](2) = std::numeric_limits<float>::quiet_NaN();
                return;
            }

            Vector3f v00, v01, v10;
            v00 = src[y][x];
            v01 = src[y][x + 1];
            v10 = src[y + 1][x];

            if (!is_nan(v00(0)) && !is_nan(v01(0)) && !is_nan(v10(0)))
            {
                Vector3f X = (v01 - v00).cross(v10 - v00);
                X.normalize();
                dest[y][x] = X;
            }
            else
            {
                dest[y][x](0) = std::numeric_limits<float>::quiet_NaN();
                dest[y][x](1) = std::numeric_limits<float>::quiet_NaN();
                dest[y][x](2) = std::numeric_limits<float>::quiet_NaN();
            }
            /*
             if (y + 1 < height(src) and x + 1 < width(src))
             {
             Vector3f X = (src[y][x + 1] - src[y][x]).cross(src[y + 1][x] - src[y][x]);
             X.normalize();
             dest[y][x] = X;
             }
             else
             {
             dest[y][x] = src[y][x] / src[y][x].norm();
             }
             */
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
const cvpr_tum::TsdfVolume& cvpr_tum::kfusionCPU::volume() const
{
    return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::TsdfVolume& cvpr_tum::kfusionCPU::volume()
{
    return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3f cvpr_tum::kfusionCPU::rodrigues2(const Eigen::Matrix3f& matrix)
{
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

    double rx = R(2, 1) - R(1, 2);
    double ry = R(0, 2) - R(2, 0);
    double rz = R(1, 0) - R(0, 1);

    double s = sqrt((rx * rx + ry * ry + rz * rz) * 0.25);
    double c = (R.trace() - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;

    double theta = acos(c);

    if (s < 1e-5)
    {
        double t;

        if (c > 0)
            rx = ry = rz = 0;
        else
        {
            t = (R(0, 0) + 1) * 0.5;
            rx = sqrt(std::max(t, 0.0));
            t = (R(1, 1) + 1) * 0.5;
            ry = sqrt(std::max(t, 0.0)) * (R(0, 1) < 0 ? -1.0 : 1.0);
            t = (R(2, 2) + 1) * 0.5;
            rz = sqrt(std::max(t, 0.0)) * (R(0, 2) < 0 ? -1.0 : 1.0);

            if (fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry * rz > 0))
                rz = -rz;
            theta /= sqrt(rx * rx + ry * ry + rz * rz);
            rx *= theta;
            ry *= theta;
            rz *= theta;
        }
    }
    else
    {
        double vth = 1 / (2 * s);
        vth *= theta;
        rx *= vth;
        ry *= vth;
        rz *= vth;
    }
    return Eigen::Vector3d(rx, ry, rz).cast<float> ();
}
