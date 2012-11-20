#include "kfusionCPU/kfusionCPU.h"
#include <Eigen/Core>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
cvpr_tum::kfusionCPU::kfusionCPU(int rows, int cols) :
    rows_(rows), cols_(cols), global_time_(0), integration_metric_threshold_(0.f)
{
    const Vector3f volume_size = Vector3f::Constant(VOLUME_SIZE);
    const Eigen::Vector3i volume_resolution(VOLUME_X, VOLUME_Y, VOLUME_Z);

    tsdf_volume_ = TsdfVolume::Ptr(new TsdfVolume(volume_resolution));
    tsdf_volume_->setSize(volume_size);

    setDepthIntrinsics(525.f, 525.f); // default values, can be overwritten

    init_Rcam_ = Eigen::Matrix3f::Identity();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
    init_tcam_ = volume_size * 0.5f - Vector3f(0, 0, volume_size(2) / 2 * 1.2f);

    const int iters[] = {10, 5, 4};
    std::copy(iters, iters + LEVELS, icp_iterations_);

    allocateMaps(rows_, cols_);

    rmats_.reserve(30000);
    tvecs_.reserve(30000);

    reset();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::allocateMaps(int rows, int cols)
{
    //  depths_curr_.resize (LEVELS);

    for (int i = 0; i < LEVELS; ++i)
    {
        int pyr_rows = rows >> i;
        int pyr_cols = cols >> i;

        depth_pyramid_[i].create(pyr_rows, pyr_cols, raw_depth_map_encoding_);
        boost::array<int, 2> shape = { {pyr_rows, pyr_cols}};
        vertex_pyramid_curr_[i].resize(shape);//boost::extents[pyr_rows][pyr_cols]);
        normal_pyramid_curr_[i].resize(shape);//resize(boost::extents[pyr_rows][pyr_cols]);
        vertex_pyramid_prev_[i].resize(shape);//boost::extents[pyr_rows][pyr_cols]);
        normal_pyramid_prev_[i].resize(shape);//resize(boost::extents[pyr_rows][pyr_cols]);
    }

}

void cvpr_tum::kfusionCPU::reset()
{
    global_time_ = 0;
    rmats_.clear();
    tvecs_.clear();

    rmats_.push_back(init_Rcam_);
    tvecs_.push_back(init_tcam_);

    //    tsdf_volume_->reset();
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
        cv::bilateralFilter(raw_depth, depth_pyramid_[0], D, sigma_colour, sigma_space);

        // 3. Compute depth pyramid.
        for (size_t i = 1; i < LEVELS; i++)
            cv::pyrDown(depth_pyramid_[i - 1], depth_pyramid_[i]);

        // 4. Compute vertex & normal pyramid.
        for (size_t i = 0; i < LEVELS; i++)
        {
            // 4.1 Populate the vertex map for the current level.
            create_vertex_map(intr(i), depth_pyramid_[i], vertex_pyramid_curr_[i]);

            // 4.2 Populate the normal map for the current level.
            create_normal_map(vertex_pyramid_curr_[i], normal_pyramid_curr_[i]);
        }
    }

    // 5. ICP
    {
        ScopeTime time("first time icp ...");
        //can't perform more on first frame
        if (global_time_ == 0)
        {
            Matrix3frm init_Rcam = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
            Vector3f init_tcam = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

            Matrix3frm init_Rcam_inv = init_Rcam.inverse();

            tsdf_volume_->integrate(raw_depth_map_, intr, init_Rcam_inv, init_tcam);

            vertex_pyramid_prev_ = vertex_pyramid_curr_; //TODO or it should be transformed?
            normal_pyramid_prev_ = normal_pyramid_curr_;

            ++global_time_;
            return (false);
        }
    }

    Matrix3frm Rprev = rmats_[global_time_ - 1]; //  [Ri|ti] - pose of camera, i.e.
    Vector3f tprev = tvecs_[global_time_ - 1]; //  transfrom from camera to global coo space for (i-1)th camera pose
    Matrix3frm Rprev_inv = Rprev.inverse(); //Rprev.t();

    Matrix3frm Rcurr = Rprev; // tranform to global coo for ith camera pose
    Vector3f tcurr = tprev;

    Eigen::Affine3f CameraTaff;
    CameraTaff.linear() = Rcurr;
    CameraTaff.translation() = tcurr;

    {
        ScopeTime time("icp-all");
        for (int level_index = LEVELS - 1; level_index >= 0; --level_index)
        {
            int iter_num = icp_iterations_[level_index];

            VertexMap vmap_curr = vertex_pyramid_curr_[level_index];
            NormalMap nmap_curr = normal_pyramid_curr_[level_index];
            VertexMap vmap_prev = vertex_pyramid_prev_[level_index];
            NormalMap nmap_prev = normal_pyramid_prev_[level_index];

            for (int iter = 0; iter < iter_num; ++iter)
            {
                Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
                Eigen::Matrix<double, 6, 1> b;

                //estimateCombined(Rcurr, tcurr, vmap_curr, nmap_curr, Rprev_inv, device_tprev, intr(level_index),
                //                 vmap_prev, nmap_prev, A.data(), b.data());

                //checking nullspace
                double det = A.determinant();

                if (fabs(det) < 1e-15) // || nan
                {
                    reset();
                    return (false);
                }

                Eigen::Matrix<float, 6, 1> result = A.llt().solve(b).cast<float> ();
                //Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

                float alpha = result(0);
                float beta = result(1);
                float gamma = result(2);

                Eigen::Matrix3f Rinc = (Eigen::Matrix3f)Eigen::AngleAxisf(gamma, Vector3f::UnitZ())
                        * Eigen::AngleAxisf(beta, Vector3f::UnitY()) * Eigen::AngleAxisf(alpha, Vector3f::UnitX());
                Vector3f tinc = result.tail<3> ();

                //compose
                tcurr = Rinc * tcurr + tinc;
                Rcurr = Rinc * Rcurr;
            }

        }

        //save tranform
        rmats_.push_back(Rcurr);
        tvecs_.push_back(tcurr);
    }

    Rprev = rmats_[global_time_ - 1];
    tprev = tvecs_[global_time_ - 1];

    Rcurr = rmats_.back();
    tcurr = tvecs_.back();
    Matrix3frm Rcurr_inv = Rcurr.inverse();

    // Integration check - We do not integrate volume if camera does not move.
    float rnorm = rodrigues2(Rcurr.inverse() * Rprev).norm();
    float tnorm = (tcurr - tprev).norm();
    const float alpha = 1.f;
    bool integrate = (rnorm + alpha * tnorm) / 2 >= integration_metric_threshold_;

    if (integrate)
    {
        //ScopeTime time("tsdf");
        tsdf_volume_->integrate(raw_depth_map_, intr, Rcurr_inv, tcurr);
    }

    ++global_time_;
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
            if (z != 0)
            {
                dest[y][x].x = z * (x - cx) / fx;
                dest[y][x].y = z * (y - cy) / fy;
                dest[y][x].z = z;
            }
            else
            {
                dest[y][x].x = std::numeric_limits<float>::quiet_NaN();
                dest[y][x].y = std::numeric_limits<float>::quiet_NaN();
                dest[y][x].z = std::numeric_limits<float>::quiet_NaN();
            }
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
            if (y + 1 < height(src) and x + 1 < width(src))
            {
                point_3d X = (src[y][x + 1] - src[y][x]) * (src[y + 1][x] - src[y][x]);
                X /= X.norm();
                dest[y][x] = X;
            }
            else
            {
                dest[y][x] = src[y][x] / src[y][x].norm();
            }
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
