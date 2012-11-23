#include <ros/ros.h> // for ros::init etc.
#include <cv_bridge/cv_bridge.h> // for CvImages
#include <image_transport/image_transport.h> // for sensor_msgs::Image
#include <sensor_msgs/image_encodings.h>
#include "kfusionCPU/kfusionCPU.h" // for kinect_fusion
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/image_viewer.h>

using namespace cvpr_tum;

struct ImageView
{
    ImageView()
    {
        viewerDepth_.setWindowTitle("Kinect Depth stream");
        viewerDepth_.setPosition(700, 50);
        viewerDepth_.setSize(640, 480);
    }

    void showGeneratedDepth(kfusionCPU& kfusion)
    {
        cv::Mat fdm(kfusion.get_raw_depth_map());
        viewerDepth_.showFloatImage(reinterpret_cast<float*> (fdm.data), fdm.cols, fdm.rows, 0, 2000, true,
                                    "raw_depth_image");
    }

    pcl::visualization::ImageViewer viewerDepth_;

    kfusionCPU::DepthMap generated_depth_;
};

struct SceneCloudView
{
    SceneCloudView() :
        cloud_viewer_("Scene Cloud Viewer")
    {
        cloud_ptr_ = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);

        cloud_viewer_.setBackgroundColor(0, 0, 0);
        cloud_viewer_.addCoordinateSystem(1.0);
        cloud_viewer_.initCameraParameters();
        cloud_viewer_.setPosition(20, 50);
        cloud_viewer_.setSize(640, 480);
        cloud_viewer_.camera_.clip[0] = 0.01;
        cloud_viewer_.camera_.clip[1] = 10.01;
    }

    void show(kfusionCPU& kfusion)
    {
        //kfusionCPU::VertexMap vmap = kfusion.get_vertex_map();
        TsdfVolume vol = kfusion.volume();

        cloud_ptr_->points.clear();
        //for(auto i = vmap.origin(); i < (vmap.origin() + vmap.num_elements()); ++i)
        //        cloud_ptr_->points.push_back( pcl::PointXYZ((*i)(0), (*i)(1), (*i)(2)) );

        for (int x = 0; x < 128; ++x)
        {
            for (int y = 0; y < 128; ++y)
            {
                for (int z = 0; z < 128; ++z)
                {
                    if (fabs(vol.getTSDFValue(Eigen::Vector3f(x * 3 / 128, y * 3 / 128, z * 3 / 128))) < 0.02)
                    {
                        cloud_ptr_->points.push_back(pcl::PointXYZ(x, y, z));
                    }
                }
            }
        }
        cloud_ptr_->width = (int)cloud_ptr_->points.size();
        cloud_ptr_->height = 1;
        cloud_viewer_.removeAllPointClouds();
        cloud_viewer_.addPointCloud<pcl::PointXYZ> (cloud_ptr_);
    }

    void clearClouds(bool print_message = false)
    {
        cloud_viewer_.removeAllPointClouds();
        cloud_ptr_->points.clear();
        if (print_message)
            cout << "Clouds/Meshes were cleared" << endl;
    }

    pcl::visualization::PCLVisualizer cloud_viewer_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ptr_;
};

class kfusionCpuApp
{

public:

    kfusionCpuApp(float vsz)
    {
        // setup publishers
        std::string raw_depth_topic = nh_.resolveName("/kfusionCPU/depth/image");
        std::string filtered_depth_topic = nh_.resolveName("/kfusionCPU/depth/image_filtered");
        uint32_t queue_size = 1;
        raw_depth_publisher_ = nh_.advertise<sensor_msgs::Image> (raw_depth_topic, queue_size);
        filtered_depth_publisher_ = nh_.advertise<sensor_msgs::Image> (filtered_depth_topic, queue_size);

        // setup subscribers
        sub_ = nh_.subscribe("/camera/depth/image", 3, &kfusionCpuApp::process_image_callback, this);

        //Init Kfusion
        Eigen::Vector3f volume_size = Eigen::Vector3f::Constant(vsz/*meters*/);
        kfusionCPU_.volume().setSize(volume_size);

        Eigen::Matrix3f R = Eigen::Matrix3f::Identity(); // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
        Eigen::Vector3f t = volume_size * 0.5f - Eigen::Vector3f(0, 0, volume_size(2) / 2 * 1.2f);

        Eigen::Affine3f pose = Eigen::Translation3f(t) * Eigen::AngleAxisf(R);

        kfusionCPU_.setInitalCameraPose(pose);
    }

    ~kfusionCpuApp()
    {
    }

    /**
     * Convert a given depth map to a ROS image, which can then be published and
     * displayed on the screen. This is mainly useful for debugging.
     * @param map The map to convert to a ROS image.
     * @return A ROS Image containing the map data.
     */
    sensor_msgs::ImagePtr map_to_image(const cv::Mat& map, std::string& encoding)
    {
        cv_bridge::CvImagePtr cv_image(new cv_bridge::CvImage);
        cv_image->image = map;
        cv_image->encoding = encoding;
        return cv_image->toImageMsg();
    }

    void process_image_callback(const sensor_msgs::ImageConstPtr& msg)
    {
        // 1. Convert the ROS image into an OpenCV image
        ROS_DEBUG("Converting ROS image to OpenCV image...");
        cv_bridge::CvImagePtr raw_depth_map = cv_bridge::toCvCopy(msg, msg->encoding);
        ROS_DEBUG("Done!");

        // 1. Run KinectFusion on the image.
        ROS_INFO("Running KinectFusion...");
        kfusionCPU_(raw_depth_map->image);
        ROS_INFO("Done!");

        // -1. Debug: publish the depth map as it was received.
        ROS_DEBUG("Publishing raw depth map read as image...");
        sensor_msgs::ImagePtr raw_image = map_to_image(kfusionCPU_.get_raw_depth_map(), raw_depth_map->encoding);
        raw_depth_publisher_.publish(raw_image);
        ROS_DEBUG("Done!");

        ROS_DEBUG("Publishing filtered depth map as image...");
        sensor_msgs::ImagePtr filtered_image = map_to_image(kfusionCPU_.filtered_depth_map(), raw_image->encoding);
        filtered_depth_publisher_.publish(filtered_image);
        ROS_DEBUG("Done!");

        image_view_.showGeneratedDepth(kfusionCPU_);

        scene_cloud_view_.show(kfusionCPU_);
        scene_cloud_view_.cloud_viewer_.spinOnce(3);
    }

    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;
    ros::Publisher raw_depth_publisher_;
    ros::Publisher filtered_depth_publisher_;

    kfusionCPU kfusionCPU_;

    ImageView image_view_;
    SceneCloudView scene_cloud_view_;
};

int main(int argc, char** argv)
{
    ROS_INFO("Initialising...");
    ros::init(argc, argv, "kinectfusion_app");
    ROS_INFO("Ready!");
    float volume_size = 3.f;
    kfusionCpuApp app(volume_size);
    ros::spin();
}

