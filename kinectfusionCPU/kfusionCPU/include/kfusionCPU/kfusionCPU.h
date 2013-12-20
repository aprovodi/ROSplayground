#ifndef INCLUDE_KFUSIONCPU_KFUSIONCPU_H_
#define INCLUDE_KFUSIONCPU_KFUSIONCPU_H_

#include <stddef.h>     // for size_t
#include <sys/types.h>  // for uintX_t
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>  // for filtering
#include <boost/array.hpp>              // for depth/vertex/normal pyramids
#include <boost/multi_array.hpp>        // for vertex and normal maps
#include <Eigen/Geometry>
#include "kfusionCPU/tsdfVolume.h"
#include "kfusionCPU/CircularBuffer.h"

namespace cvpr_tum {

/**
 * An implementation of the KinectFusion algorithm for transforming the Kinect
 * depth input into a point cloud.
 */

//#define TIME_MEASUREMENTS

const float VOLUME_SIZE = 4.0f;         // in meters
const float DISTANCE_THRESHOLD = 0.5f; // when the camera target point is farther than DISTANCE_THRESHOLD from the current cube's center, shifting occurs. In meters

class kfusionCPU {
    enum Pyramid { LEVELS = 5 };

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;
        typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> Matrix4frm;
        typedef Eigen::Vector3f Vector3f;
        typedef Eigen::Vector3i Vector3i;
        typedef Eigen::Matrix<float, 6, 1> Vector6f;

        typedef Eigen::Vector3f Point3DType;

        typedef cv::Mat DepthMap;
        typedef boost::multi_array<Point3DType, 2> VertexMap;
        typedef boost::multi_array<Point3DType, 2> NormalMap;
        typedef boost::array<DepthMap, LEVELS> DepthPyramid;
        typedef boost::array<VertexMap, LEVELS> VertexPyramid;
        typedef boost::array<NormalMap, LEVELS> NormalPyramid;

        /**
         * Build a kinect fusion object, which can be run to process a ROS
         * image using the kinect fusion algorithm.
         */

        /** \brief Constructor
         * \param[in] rows width of the input depth image
         * \param[in] cols height of the input depth image
         */
        kfusionCPU(int rows = 480, int cols = 640, int encoding = 5);

        /** \brief Destructor */
        ~kfusionCPU();

        /** \brief Performs the tracker reset to initial  state. It's used if case of camera tracking fail.
         */
        void reset();

        /** \brief Sets Depth camera intrinsics
         * \param[in] fx focal length x
         * \param[in] fy focal length y
         * \param[in] cx principal point x
         * \param[in] cy principal point y
         */
        void setDepthIntrinsics(float fx, float fy, float cx = -1, float cy = -1);

        /** \brief Sets initial camera pose relative to volume coordiante space
         * \param[in] pose Initial camera pose
         */
        void setInitalCameraPose(const Eigen::Affine3f& pose);

        /**
         * Process an image using the KinectFusion algorithm. This is the main
         * algorithm to call when using this implementation of KinectFusion.
         *
         * @param image The image message received from the Openni Kinect driver.
         */
        bool operator()(const DepthMap& image);

        /** \brief Returns pointer to the cyclical buffer structure */
        CircularBuffer* getCircularBufferStructure () { return (&cyclical_); }

        /** \brief Returns TSDF volume storage */
        const TsdfVolume& volume() const { return *tsdf_volume_; }

        /** \brief Returns TSDF volume storage */
        TsdfVolume& volume() { return *tsdf_volume_; }
        TsdfVolume::TsdfVolumePtr volume_ptr() { return tsdf_volume_; }
        TsdfVolume& first_volume();

        /** \brief Returns camera pose at given time, default the last pose
         * \param[in] time Index of frame for which camera pose is returned.
         * \return camera pose
         */
        Eigen::Affine3f getCameraPose(int time = -1) const;

        void pyrDownMedianSmooth(const cv::Mat& in, cv::Mat& out);

        inline const DepthMap get_raw_depth_map() const { return raw_depth_map_; }

        inline const DepthMap filtered_depth_map() const { return depth_pyramid_[0]; }

        inline const VertexMap get_vertex_map() const { return vertex_pyramid_[0]; }

    private:

        /** \brief Cyclical buffer object */
        CircularBuffer cyclical_;

        /** \brief Height of input depth image. */
        int rows_;
        /** \brief Width of input depth image. */
        int cols_;

        /** \brief A depth map of raw values, as taken from the input image. */
        DepthMap raw_depth_map_;

        /** \brief A depth map encoding. */
        int raw_depth_map_encoding_;

        /** \brief array with IPC iteration numbers for each pyramid level */
        int icp_iterations_[LEVELS];

        /** \brief A filtered depth map, produced using a bilinear filter on the original image. */
        DepthPyramid depth_pyramid_;

        /** \brief Vertex maps pyramid for current frame. */
        VertexPyramid vertex_pyramid_;

        /** \brief Normal maps pyramid for current frame. */
        NormalPyramid normal_pyramid_;

        /** \brief Frame counter */
        int global_time_;

        /** \brief Intrinsic parameters of depth camera. */
        float fx_, fy_, cx_, cy_;

        /** \brief Tsdf volume container. */
        TsdfVolume::TsdfVolumePtr tsdf_volume_;

        /** \brief Initial camera rotation in volume coo space. */
        Eigen::Affine3f init_Tcam_;

        /** \brief Transformation composed of the camera (SO(3)). */
        Eigen::Affine3f Transformation_;

        /** \brief Transformation in so(3). */
        Vector6f Pose_;

        /** \brief Measurement of camera displacement. */
        Vector6f cumulative_pose_;

        // data required for the bilinear filtering
//        static const float sigma_colour = 30; // mm
//        static const float sigma_space = 4.5; // pixels
//        static const int D = (6 * 2 + 1);

        /** \brief Array of camera rotation matrices for each moment of time. */
        std::vector<Eigen::Affine3f, Eigen::aligned_allocator<Eigen::Affine3f> > Tmats_;

        /** \brief Camera movement threshold. TSDF is integrated iff a camera movement metric exceedes some value. */
        const float integration_metric_threshold_;

        const float robust_statistic_coefficient_;

        const float regularization_;

        const float angular_change_max_;
        const float mov_change_max_;

        /** \brief Size of the TSDF volume in meters. */
        Eigen::Vector3f volume_size_;

        bool disable_intergration_;

        /** \brief Allocates all internal buffers.
         * \param[in] rows_arg
         * \param[in] cols_arg
         */
        void allocateMaps(int rows_arg, int cols_arg);

        /** \brief Composes Twist from vector6f ksi */
        Eigen::Matrix4f Twist(const Eigen::Matrix<float, 6, 1>& ksi);

        /** \brief Calculates camera transfromations in each iteration */
        Vector6f TrackCameraPose(void);

        /** \brief Creates a vertex map given a depth map
         * \param[in] intr camera intrinsics
         * \param[in] src given depth map
         * \param[in] dest vertex map
         */
        void create_vertex_map(const Intr& intr, const DepthMap& src, VertexMap& dest);

        /** \brief Create a normal map given a vertex map
         * \param[in] intr camera intrinsics
         * \param[in] src given depth map
         * \param[in] dest vertex map
         */
        void create_normal_map(const VertexMap& src, NormalMap& dest);

        /** \brief Integrates TSDF Volume
         * \param[in] raw_depth_map
         * \param[in] intr Camera Intrinsics
         * \param[in] Rcam_inv Rotational part
         * \param[in] tcam Translational part
         */
        void integrate(const cv::Mat& raw_depth_map, const Intr& intr, const Eigen::Matrix4f& camtoworld);

        Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);

        /** \brief helper methods for processing (or readability) */
        static size_t height(const DepthMap& map) { return map.size().height; }
        static size_t rows(const DepthMap& map) { return height(map); }
        static size_t height(const VertexMap& map) { return map.shape()[0]; }
        static size_t rows(const VertexMap& map) { return height(map); }

        static size_t width(const DepthMap& map) { return map.size().width; }
        static size_t cols(const DepthMap& map) { return width(map); }
        static size_t width(const VertexMap& map) { return map.shape()[1]; }
        static size_t cols(const VertexMap& map) { return width(map); }
};
}  // namespace cvpr_tum
#endif  // INCLUDE_KFUSIONCPU_KFUSIONCPU_H_
