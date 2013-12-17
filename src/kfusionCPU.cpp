#include "kfusionCPU/kfusionCPU.h"

#include <limits>       // for numeric_limits
#include <Eigen/StdVector>
#include <unsupported/Eigen/MatrixFunctions>
#include <opencv2/highgui/highgui.hpp>
#include "OptimizedSelfAdjointMatrix6x6f.cpp"

////////////////////////////////////////////////////////////////////////////////
cvpr_tum::kfusionCPU::kfusionCPU(int rows, int cols, int encoding)
    : cyclical_( DISTANCE_THRESHOLD, VOLUME_SIZE, TsdfVolume::VOLUME_X),
      rows_(rows), cols_(cols), raw_depth_map_encoding_(encoding),
      global_time_(0), integration_metric_threshold_(0.01f),
      robust_statistic_coefficient_(0.02f), regularization_(0.01f),
      angular_change_max_(0.1), mov_change_max_(0.3) {

    std::cout << "KFUSION CONSTRUCTOR WAS CALLED" << std::cout;

    const Vector3f volume_size = Vector3f::Constant(VOLUME_SIZE);

    volume_size_ = volume_size(0);

    const Vector3i volume_resolution(TsdfVolume::VOLUME_X, TsdfVolume::VOLUME_Y, TsdfVolume::VOLUME_Z);

    tsdf_volume_ = TsdfVolume::TsdfVolumePtr(new TsdfVolume(volume_resolution));
    tsdf_volume_->setSize(volume_size);

    cyclical_.setDistanceThreshold(DISTANCE_THRESHOLD);
    cyclical_.setVolumeSize(volume_size);

    setDepthIntrinsics(520.f, 520.f, 319.5f, 239.5f);  // default values, can be overwritten

    init_Tcam_ = Eigen::Affine3f(Eigen::Translation3f(-Eigen::Vector3f(0, 0, volume_size (2) / 2 * 1.0f)));

    const int iters[] = {0, 0, 2, 4, 8};
    std::copy(iters, iters + LEVELS, icp_iterations_);

    allocateMaps(rows_, cols_);

    Tmats_.reserve(30000);

    reset();

    // initialize cyclical buffer
    cyclical_.initBuffer(tsdf_volume_);
}

cvpr_tum::kfusionCPU::~kfusionCPU() {
    std::cout << "KFUSION DESTRUCTOR WAS CALLED" << std::cout;
    //tsdf_volume_->release();
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::allocateMaps(int rows, int cols) {
    for (std::size_t i = 0; i < LEVELS; ++i) {
        int pyr_rows = rows >> i;
        int pyr_cols = cols >> i;

        depth_pyramid_[i].create(pyr_rows, pyr_cols, raw_depth_map_encoding_);

        boost::array<int, 2> shape = { {pyr_rows, pyr_cols}};
        vertex_pyramid_[i].resize(shape);  // resize(boost::extents[pyr_rows][pyr_cols]);
        normal_pyramid_[i].resize(shape);  // resize(boost::extents[pyr_rows][pyr_cols]);
    }
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::reset() {
    global_time_ = 0;

    Tmats_.clear();
    Tmats_.push_back(init_Tcam_);

    Transformation_ = init_Tcam_;

    Pose_ = Vector6f::Zero();
    cumulative_pose_ *= 0.f;

    tsdf_volume_->reset();

    // reset cyclical buffer as well
    cyclical_.resetBuffer(tsdf_volume_);
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::setInitalCameraPose(const Eigen::Affine3f& pose) {
    init_Tcam_ = pose;
    reset();
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::setDepthIntrinsics(float fx, float fy, float cx, float cy) {
    fx_ = fx;
    fy_ = fy;
    cx_ = (cx == -1) ? cols_ / 2 - 0.5f : cx;
    cy_ = (cy == -1) ? rows_ / 2 - 0.5f : cy;
}

////////////////////////////////////////////////////////////////////////////////
bool cvpr_tum::kfusionCPU::operator()(const DepthMap& raw_depth) {
    ScopeTime time(">>> TOTAL TIME FOR ONE ITERATION");
    Intr intr(fx_, fy_, cx_, cy_);

    raw_depth_map_ = raw_depth;

    {
        ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");

        // 2. Apply bilinear filtering (using OpenCV)
        //        cv::bilateralFilter(raw_depth_map_, depth_pyramid_[0], D, sigma_colour, sigma_space);
        depth_pyramid_[0] = raw_depth_map_;

        // 3. Compute depth pyramid.
        for (size_t i = 1; i < LEVELS; i++)
            //            cv::pyrDown(depth_pyramid_[i - 1], depth_pyramid_[i]);
            pyrDownMedianSmooth(depth_pyramid_[i - 1], depth_pyramid_[i]);

        // 4. Compute vertex pyramid.
        // TODO(provodin): check if the current num of iter is 0
        for (size_t i = 0; i < LEVELS; i++)
            create_vertex_map(intr(i), depth_pyramid_[i], vertex_pyramid_[i]);
    }

    if (global_time_ > 0)
    {
        ScopeTime time(">>> Camera Tracking");
        Pose_ = TrackCameraPose();
    }

    Transformation_.matrix() = Twist(Pose_).exp() * Transformation_.matrix();
    Tmats_.push_back(Transformation_);

    ///////////////////////////////////////////////////////////////////////////////////////////
    // check if we need to shift
//    bool has_shifted = cyclical_.checkForShift(tsdf_volume_, getCameraPose(), 0.5 * volume_size_, false);
//    if(has_shifted)
//        std::cout << "SHIFTING" << std::endl;

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Integration check - We do not integrate volume if camera does not move far enought.
    cumulative_pose_ += Pose_;
    {
        ScopeTime time(">>> Volume Integration");

        bool integrate = (cumulative_pose_.norm() < integration_metric_threshold_ && global_time_ > 0);

        ///////////////////////////////////////////////////////////////////////////////////////////
        // Volume integration
        if (integrate)
        {
            Eigen::Matrix3f Rcam_inv = Transformation_.linear().inverse();
            Eigen::Vector3f tcam = Transformation_.translation();
            //tcam -= getCircularBufferStructure()->get_current_origin(); // translation (local translation = global translation - origin of cube)
            tsdf_volume_->integrate(raw_depth_map_, intr, Rcam_inv, tcam, getCircularBufferStructure() );
        }

        cumulative_pose_ *= 0.f;
        Pose_ = Pose_ * 0.f;
    }

    global_time_++;
    return (true);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix<float, 6, 1> cvpr_tum::kfusionCPU::TrackCameraPose() {
    Vector6f ksi = Vector6f::Zero();
    Vector6f ksi_prev = ksi;

    //partial derivative of SDF wrt position
    Eigen::Matrix<float, 1, 3> dSDF_dx = Eigen::Matrix<float, 1, 3>::Zero();
    //partial derivative of position wrt optimizaiton parameters
    Eigen::Matrix<float, 3, 6> dx_dksi = Eigen::Matrix<float, 3, 6>::Zero();
    //jacobian = derivative of SDF wrt ksi (chain rule)
    Eigen::Matrix<float, 1, 6> J = Eigen::Matrix<float, 1, 6>::Zero();

    OptimizedSelfAdjointMatrix6x6f osam;

    for (int level_index = LEVELS - 1; level_index >= 0; --level_index) {
        int iter_num = icp_iterations_[level_index];

        VertexMap &vmap_curr = vertex_pyramid_[level_index];

        for (int iter = 0; iter < iter_num; ++iter) {
            Eigen::Matrix4f camToWorld = Twist(ksi).exp() * Transformation_.matrix();
            Eigen::Matrix3f Rcam = camToWorld.topLeftCorner<3, 3> ();
            Eigen::Vector3f tcam = camToWorld.topRightCorner<3, 1> ();
            //tcam -= getCircularBufferStructure ()->get_current_origin();

            Eigen::Matrix<float, 6, 6> A = Eigen::Matrix<float, 6, 6>::Zero();
            Eigen::Matrix<float, 6, 1> b = Eigen::Matrix<float, 6, 1>::Zero();
            osam.setZero();

            //for (size_t i = 0; i < vmap_curr.shape()[0]; i++)
            //for (size_t j = 0; j < vmap_curr[0].shape()[1]; j++)
            for (VertexMap::element* i = vmap_curr.origin(); i < (vmap_curr.origin() + vmap_curr.num_elements()); ++i) {
                const Eigen::Vector3f& v = *i;
                //tranform to global coo space

                //TODO : exclude nan's'
                if (is_nan(v(2)))
                    continue;

                Eigen::Vector3f v_g = Rcam * v + tcam;

                //if (!tsdf_volume_->validGradient(v_g)) continue;

                float D = tsdf_volume_->getInterpolatedTSDFValue(v_g);

                // opt: make methods inline or store value in temporary?!
                if (fabs(D - tsdf_volume_->getPositiveTsdfTruncDist()) < std::numeric_limits<float>::epsilon()
                        || fabs(D - tsdf_volume_->getNegativeTsdfTruncDist()) < std::numeric_limits<float>::epsilon())
                    continue;

                // partial derivative of SDF wrt position
                dSDF_dx = tsdf_volume_->getTSDFGradient(v_g);
                //partial derivative of position wrt optimizaiton parameters

                // opt: compute the 6 values in J analytically as expressions of dSDF_dx and v_g and hard code the expressions here! Saves matrix multiplication!
                dx_dksi << 0, v_g(2), -v_g(1), 1, 0, 0, -v_g(2), 0, v_g(0), 0, 1, 0, v_g(1), -v_g(0), 0, 0, 0, 1;

                // jacobian = derivative of SDF wrt ksi (chain rule)
                J = dSDF_dx * dx_dksi;

                const float c = robust_statistic_coefficient_ * tsdf_volume_->getPositiveTsdfTruncDist();
                float huber = fabs(D) < c ? 1.0 : c / fabs(D);

                // Gauss - Newton approksimation to hessian
                // A += huber * J.transpose() * J;
                osam.rankUpdate(J.transpose(), huber);
                b += huber * J.transpose() * D;
            }

            osam.toEigen(A);

            double scaling = 1 / A.maxCoeff();
            A *= scaling;
            b *= scaling;

            A = A + (regularization_) * Eigen::MatrixXf::Identity(6, 6);
            ksi = ksi - A.llt().solve(b).cast<float> ();

            Vector6f Change = ksi - ksi_prev;
            double Cnorm = Change.norm();
            ksi_prev = ksi;
            if (Cnorm < 0.0001)
                break;
        }  // iterations
    }  // levels

    if (is_nan(ksi.sum()))
        ksi = Vector6f::Zero();
    return ksi;
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::create_vertex_map(const Intr& intr, const DepthMap& src, VertexMap& dest) {
    // get intrinsic parameters
    float fx = intr.fx, cx = intr.cx;
    float fy = intr.fy, cy = intr.cy;

    // for each pixel in the old depth map...
    for (size_t y = 0; y < height(src); y++) {
        for (size_t x = 0; x < width(src); x++) {
            // get the depth reading at this point
            float z = src.at<float> (y, x);  // convert mm --> m if needed

            // if we have some depth reading...
            if (z != 0) {
                dest[y][x](0) = z * (x - cx) / fx;
                dest[y][x](1) = z * (y - cy) / fy;
                dest[y][x](2) = z;
            }
            else {
                dest[y][x](0) = std::numeric_limits<float>::quiet_NaN();
                dest[y][x](1) = std::numeric_limits<float>::quiet_NaN();
                dest[y][x](2) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
void cvpr_tum::kfusionCPU::create_normal_map(const VertexMap& src, NormalMap& dest) {
    // for each vertex in the vertex map...
    for (size_t y = 0; y < height(src); y++) {
        for (size_t x = 0; x < width(src); x++) {
            if (x == width(src) - 1 || y == height(src) - 1) {
                dest[y][x](0) = std::numeric_limits<float>::quiet_NaN();
                dest[y][x](1) = std::numeric_limits<float>::quiet_NaN();
                dest[y][x](2) = std::numeric_limits<float>::quiet_NaN();
                return;
            }

            Vector3f v00, v01, v10;
            v00 = src[y][x];
            v01 = src[y][x + 1];
            v10 = src[y + 1][x];

            if (!is_nan(v00(0)) && !is_nan(v01(0)) && !is_nan(v10(0))) {
                Vector3f X = (v01 - v00).cross(v10 - v00);
                X.normalize();
                dest[y][x] = X;
            }
            else {
                dest[y][x](0) = std::numeric_limits<float>::quiet_NaN();
                dest[y][x](1) = std::numeric_limits<float>::quiet_NaN();
                dest[y][x](2) = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Vector3f cvpr_tum::kfusionCPU::rodrigues2(const Eigen::Matrix3f& matrix) {

    Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);
    Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

    double rx = R(2, 1) - R(1, 2);
    double ry = R(0, 2) - R(2, 0);
    double rz = R(1, 0) - R(0, 1);

    double s = sqrt((rx * rx + ry * ry + rz * rz) * 0.25);
    double c = (R.trace() - 1) * 0.5;
    c = c > 1. ? 1. : c < -1. ? -1. : c;

    double theta = acos(c);

    if (s < 1e-5) {
        double t;

        if (c > 0)
            rx = ry = rz = 0;
        else {
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
    } else {
        double vth = 1 / (2 * s);
        vth *= theta;
        rx *= vth;
        ry *= vth;
        rz *= vth;
    }
    return Eigen::Vector3d(rx, ry, rz).cast<float> ();
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f cvpr_tum::kfusionCPU::getCameraPose(int time) const {
    if (time > (int)Tmats_.size() || time < 0)
        time = Tmats_.size() - 1;

    return (Tmats_[time]);
}

void cvpr_tum::kfusionCPU::pyrDownMedianSmooth(const cv::Mat& in, cv::Mat& out) {
    out.create(cv::Size(in.size().width / 2, in.size().height / 2), in.type());

    cv::Mat in_smoothed;
    cv::medianBlur(in, in_smoothed, 3);

    for (int y = 0; y < out.rows; ++y)
        for (int x = 0; x < out.cols; ++x)
            out.at<float>(y, x) = in_smoothed.at<float>(y * 2, x * 2);
}

////////////////////////////////////////////////////////////////////////////////
Eigen::Matrix4f cvpr_tum::kfusionCPU::Twist(const Eigen::Matrix<float, 6, 1>& ksi) {
    Eigen::Matrix4f M;
    M << 0.f, -ksi(2), ksi(1), ksi(3), ksi(2), 0.f, -ksi(0), ksi(4), -ksi(1), ksi(0), 0.f, ksi(5), 0.f, 0.f, 0.f, 0.f;
    return M;
}
