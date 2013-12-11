#include <ros/ros.h> // for ros::init etc.
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/subscriber.h>

#include <pcl17/visualization/pcl_visualizer.h>
#include <pcl17/visualization/image_viewer.h>
#include <pcl17/ros/conversions.h>
#include <pcl17_ros/point_cloud.h>
#include <tf_conversions/tf_eigen.h>
#include <boost/foreach.hpp>
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include "kfusionCPU/kfusionCPU.h"
#include "kfusionCPU/tsdfVolume.h"
#include "kfusionCPU_msgs/Volume.h"
#include "kfusionCPU_visualization/SceneCloudView.h"

using namespace cvpr_tum;

struct kfusionCpuVis
{
    kfusionCpuVis(ros::NodeHandle& nh, ros::NodeHandle& nh_private) :
        nh_(nh), nh_private_(nh_private),
        volume_sub_(nh, "/kfusionCPU/tsdf_volume", 1),
        cam_translation_sub_(nh, "/kfusionCPU/transformations", 1),
        sync( MySyncPolicy( 10 ), volume_sub_, cam_translation_sub_),
        volume_resolution_(TsdfVolume::VOLUME_X, TsdfVolume::VOLUME_Y, TsdfVolume::VOLUME_Z),
        world_size_(3.0), scene_cloud_view_()
    {
        tsdf_volume_ = TsdfVolume::TsdfVolumePtr(new TsdfVolume(volume_resolution_));

        sync.registerCallback( boost::bind( &kfusionCpuVis::process_callback, this, _1, _2 ) );

        // setup publishers
        std::string raw_depth_topic = nh_.resolveName("/kfusionCPU/depth/image");
        std::string filtered_depth_topic = nh_.resolveName("/kfusionCPU/depth/image_filtered");
        uint32_t queue_size = 1;
        raw_depth_publisher_ = nh_.advertise<sensor_msgs::Image> (raw_depth_topic, queue_size);
        filtered_depth_publisher_ = nh_.advertise<sensor_msgs::Image> (filtered_depth_topic, queue_size);
        reset_publisher_ = nh_.advertise<std_msgs::Bool> ("/reset", queue_size);

        // setup subscribers
        cur_pc_sub_ = nh_.subscribe("/camera/depth_registered/points", 1, &kfusionCpuVis::process_cur_pc_callback, this);

        scene_cloud_view_.registerKeyboardCallback(keyboard_callback, (void*)this);
    }

    ~kfusionCpuVis() { }

    void process_callback(const kfusionCPU_msgs::VolumeConstPtr& msg, const geometry_msgs::PoseWithCovarianceStampedConstPtr& pose)
    {
        world_size_(0) = msg->world_size.x;
        world_size_(1) = msg->world_size.y;
        world_size_(2) = msg->world_size.z;

        tsdf_volume_->setSize(world_size_.cast<float>());
        tsdf_volume_->copyFrom((float *)msg->data.data(), msg->num_voxels);
        {
            ScopeTime time(">>> VISUALIZATION.......................");

            tf::Transform tmp_tf;

            tf::poseMsgToTF(pose->pose.pose, tmp_tf);
            tf::TransformTFToEigen(tmp_tf, cam_pose_);
            cam_pose_.translation() += world_size_ / 2;

            scene_cloud_view_.addCoordinateSystem(cam_pose_, false);
            scene_cloud_view_.updateCamPCloud(cam_pose_.translation());

            scene_cloud_view_.renderScene(tsdf_volume_, true);
        }

    }

    void process_cur_pc_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
    {
        scene_cloud_view_.update_current_cloud(msg, cam_pose_, world_size_);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    static void keyboard_callback(const pcl17::visualization::KeyboardEvent &e, void *cookie)
    {
        kfusionCpuVis* app = reinterpret_cast<kfusionCpuVis*> (cookie);

        int key = e.getKeyCode();

        if (e.keyUp())
            switch (key)
            {
            case (int)'r':
            case (int)'R':
            {
                std_msgs::Bool true_msg;
                true_msg.data = true;
                app->reset_publisher_.publish(true_msg);
                app->scene_cloud_view_.reset();
                break;
            }
            case (int)'b':
            case (int)'B':
                app->scene_cloud_view_.toggleCube(app->world_size_.cast<float>());
                break;
            case (int)'c':
            case (int)'C':
                app->scene_cloud_view_.toggleCamPose();
                break;
            case (int)'v':
            case (int)'V':
                app->scene_cloud_view_.toggleCurView();
                break;
            default:
                break;
            }
    }

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    typedef message_filters::Subscriber< kfusionCPU_msgs::Volume > VolumeSubscriber;
    VolumeSubscriber volume_sub_;
    typedef message_filters::Subscriber< geometry_msgs::PoseWithCovarianceStamped > CPoseSubscriber;
    CPoseSubscriber cam_translation_sub_;
    typedef message_filters::sync_policies::ApproximateTime< kfusionCPU_msgs::Volume, geometry_msgs::PoseWithCovarianceStamped > MySyncPolicy;
    message_filters::Synchronizer< MySyncPolicy > sync;

    ros::Publisher pub_;
    ros::Subscriber cur_pc_sub_;
    ros::Publisher raw_depth_publisher_;
    ros::Publisher filtered_depth_publisher_;
    ros::Publisher reset_publisher_;

    TsdfVolume::TsdfVolumePtr tsdf_volume_;

    const Eigen::Vector3i volume_resolution_;
    Eigen::Vector3d world_size_;
    Eigen::Affine3d cam_pose_;

    SceneCloudView scene_cloud_view_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kinectfusion_vis");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    kfusionCpuVis app(nh, nh_private);
    ROS_INFO("kfusion algorithm visualization started...");

    ros::spin();
    return 0;
}

struct ImageView
{
    ImageView()
    {
        viewerDepth_.setWindowTitle("Kinect Depth stream");
        viewerDepth_.setPosition(700, 50);
        viewerDepth_.setSize(1280, 960);
    }

    void showGeneratedDepth(kfusionCPU& kfusion)
    {
        cv::Mat fdm(kfusion.get_raw_depth_map());
        viewerDepth_.showFloatImage(reinterpret_cast<float*> (fdm.data), fdm.cols, fdm.rows, 0, 2000, true,
                                    "raw_depth_image");
    }

    pcl17::visualization::ImageViewer viewerDepth_;

    kfusionCPU::DepthMap generated_depth_;
};
