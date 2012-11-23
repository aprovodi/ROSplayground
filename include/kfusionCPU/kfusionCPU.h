#ifndef __KFUSIONCPU_H__
#define __KFUSIONCPU_H__

#include <limits> // for numeric_limits
#include <stddef.h> // for size_t
#include <sys/types.h> // for uintX_t
#include <opencv2/imgproc/imgproc.hpp> // for filtering
#include <opencv2/contrib/contrib.hpp>
#include <boost/array.hpp> // for depth/vertex/normal pyramids
#include <boost/multi_array.hpp> // for vertex and normal maps
#include "kfusionCPU/tdsfVolume.h"
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace cvpr_tum
{
class point_3d
{
public:

    float x;
    float y;
    float z;

    point_3d& operator=(const point_3d& other)
    {
        x = other.x;
        y = other.y;
        z = other.z;
        return *this;
    }

    point_3d operator-(const point_3d& other) const
    {
        point_3d tmp;
        tmp.x = x - other.x;
        tmp.y = y - other.y;
        tmp.z = z - other.z;
        return tmp;
    }

    // perform a CROSS product (NOT dot product)
    point_3d operator*(const point_3d& other) const
    {
        point_3d tmp;
        tmp.x = y * other.z - z * other.y;
        tmp.y = z * other.x - x * other.z;
        tmp.z = x * other.y - y * other.x;
        return tmp;
    }

    point_3d& operator/=(float scalar)
    {
        x /= scalar;
        y /= scalar;
        x /= scalar;
        return *this;
    }

    point_3d operator/(float scalar) const
    {
        point_3d tmp;
        tmp.x = x / scalar;
        tmp.y = y / scalar;
        tmp.z = z / scalar;
        return tmp;
    }

    float norm() const
    {
        return sqrt(x * x + y * y + z * z);
    }
};

/** \brief Camera intrinsics structure
 */
class Intr
{
public:
    float fx, fy, cx, cy;
    Intr()
    {
    }
    Intr(float fx_, float fy_, float cx_, float cy_) :
        fx(fx_), fy(fy_), cx(cx_), cy(cy_)
    {
    }

    Intr operator()(int level_index) const
    {
        int div = 1 << level_index;
        return (Intr(fx / div, fy / div, cx / div, cy / div));
    }

    friend inline std::ostream&
    operator <<(std::ostream& os, const Intr& intr)
    {
        os << "([f = " << intr.fx << ", " << intr.fy << "] [cp = " << intr.cx << ", " << intr.cy << "])";
        return (os);
    }
};

/**
 * An implementation of the KinectFusion algorithm for transforming the Kinect
 * depth input into a point cloud.
 */

const float VOLUME_SIZE = 3.0f; // in meters

class kfusionCPU
{
    enum Pyramid
    {
        LEVELS = 3
    };

    //should be multiple of 32
    enum VolumeResolution
    {
        VOLUME_X = 512, VOLUME_Y = 512, VOLUME_Z = 512
    };

public:

    typedef cv::Mat DepthMap;
    typedef boost::multi_array<point_3d, 2> VertexMap;
    typedef boost::multi_array<point_3d, 2> NormalMap;
    typedef boost::array<DepthMap, LEVELS> DepthPyramid;
    typedef boost::array<VertexMap, LEVELS> VertexPyramid;
    typedef boost::array<NormalMap, LEVELS> NormalPyramid;

    typedef Eigen::Matrix<float, 3, 3, Eigen::RowMajor> Matrix3frm;
    typedef Eigen::Vector3f Vector3f;

    /**
     * Build a kinect fusion object, which can be run to process a ROS
     * image using the kinect fusion algorithm.
     */
    kfusionCPU(int rows = 480, int cols = 640);

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

    /** \brief Returns TSDF volume storage */
    const TsdfVolume& volume() const;

    /** \brief Returns TSDF volume storage */
    TsdfVolume& volume();

    inline const DepthMap get_raw_depth_map() const
    {
        return raw_depth_map_;
    }
    inline const DepthMap filtered_depth_map() const
    {
        return depth_pyramid_[0];
    }
    inline const VertexMap get_vertex_map() const
    {
        return vertex_pyramid_curr_[0];
    }

private:

    /** \brief A depth map of raw values, as taken from the input image. */
    DepthMap raw_depth_map_;
    int raw_depth_map_encoding_;

    /** \brief A filtered depth map, produced using a bilinear filter on the original image. */
    DepthPyramid depth_pyramid_;

    /** \brief Vertex maps pyramid for current frame. */
    VertexPyramid vertex_pyramid_curr_;

    /** \brief Normal maps pyramid for current frame. */
    NormalPyramid normal_pyramid_curr_;

    /** \brief Vertex maps pyramid for previous frame. */
    VertexPyramid vertex_pyramid_prev_;

    /** \brief Normal maps pyramid for previous frame. */
    NormalPyramid normal_pyramid_prev_;

    /** \brief Height of input depth image. */
    int rows_;
    /** \brief Width of input depth image. */
    int cols_;
    /** \brief Frame counter */
    int global_time_;

    /** \brief Intrinsic parameters of depth camera. */
    float fx_, fy_, cx_, cy_;

    /** \brief Tsdf volume container. */
    TsdfVolume::Ptr tsdf_volume_;

    /** \brief Initial camera rotation in volume coo space. */
    Matrix3frm init_Rcam_;

    /** \brief Initial camera position in volume coo space. */
    Vector3f init_tcam_;

    /** \brief Transformation composed of Rcam and tcam. */
    Eigen::Matrix4d transformation_;

    /** \brief array with IPC iteration numbers for each pyramid level */
    int icp_iterations_[LEVELS];

    // data required for the bilinear filtering
    static const float sigma_colour = 30; // mm
    static const float sigma_space = 4.5; // pixels
    static const int D = (6 * 2 + 1);

    /** \brief Array of camera rotation matrices for each moment of time. */
    std::vector<Matrix3frm> rmats_;

    /** \brief Array of camera translations for each moment of time. */
    std::vector<Vector3f> tvecs_;

    /** \brief Camera movement threshold. TSDF is integrated iff a camera movement metric exceedes some value. */
    float integration_metric_threshold_;

    /** \brief Allocates all internal buffers.
     * \param[in] rows_arg
     * \param[in] cols_arg
     */
    void allocateMaps(int rows_arg, int cols_arg);

    /** \brief Performs the tracker reset to initial  state. It's used if case of camera tracking fail.
     */
    void reset();

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

    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);

    /** \brief helper methods for processing (or readability) */
    static size_t height(const DepthMap& map)
    {
        return map.size().height;
    }
    static size_t rows(const DepthMap& map)
    {
        return height(map);
    }
    static size_t height(const VertexMap& map)
    {
        return map.shape()[0];
    }
    static size_t rows(const VertexMap& map)
    {
        return height(map);
    }

    static size_t width(const DepthMap& map)
    {
        return map.size().width;
    }
    static size_t cols(const DepthMap& map)
    {
        return width(map);
    }
    static size_t width(const VertexMap& map)
    {
        return map.shape()[1];
    }
    static size_t cols(const VertexMap& map)
    {
        return width(map);
    }

};

/******************\
 * HELPER METHODS *
 ******************/
struct ScopeTime
{
    const char* name;
    cv::TickMeter tm;
    ScopeTime(const char *name_) :
        name(name_)
    {
        tm.start();
    }
    ~ScopeTime()
    {
        tm.stop();
        std::cout << "Time(" << name << ") = " << tm.getTimeMilli() << "ms" << std::endl;
    }
};
}
#endif /* __KFUSIONCPU_H__ */