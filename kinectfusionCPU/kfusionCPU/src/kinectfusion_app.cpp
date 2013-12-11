#include <ros/ros.h>  // for ros::init etc.
#include <cv_bridge/cv_bridge.h>  // for CvImages
#include <image_transport/image_transport.h>  // for sensor_msgs::Image
#include <sensor_msgs/image_encodings.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <std_msgs/Bool.h>
#include <tf_conversions/tf_eigen.h>

#include "kfusionCPU/kfusionCPU.h"  // for kinect_fusion
#include "kfusionCPU_msgs/Volume.h"

struct kfusionCpuApp {
    kfusionCpuApp(ros::NodeHandle& nh, ros::NodeHandle& nh_private) :
        nh_(nh),
        nh_private_(nh_private)
    {
        // setup subscribers
        depth_image_sub_ = nh_.subscribe("/camera/depth/image_raw", 1, &kfusionCpuApp::process_image_callback, this);
        reset_sub_ = nh_.subscribe("/reset", 1, &kfusionCpuApp::reset_kfusion, this);

        // setup publishers
        uint32_t queue_size = 1;
        std::string topic_volume = nh_.resolveName("/kfusionCPU/tsdf_volume");
        volume_pub_ = nh_.advertise<kfusionCPU_msgs::Volume> (topic_volume, queue_size);

        std::string topic_translation = nh_.resolveName("/kfusionCPU/transformations");
        pose_pub_ = nh_.advertise<geometry_msgs::PoseWithCovarianceStamped> (topic_translation, queue_size);

        // init Kfusion
        nh_.param("world_size", world_size_, 3.0);
        Eigen::Vector3f volume_size = Eigen::Vector3f::Constant(world_size_/*meters*/);
        kfusionCPU_.volume().setSize(volume_size);

        Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
        Eigen::Vector3f t = -Eigen::Vector3f(0, 0, volume_size (2) / 2 * 1.0f);

        Eigen::Affine3f pose = Eigen::Translation3f(t) * Eigen::AngleAxisf(R);
        kfusionCPU_.setInitalCameraPose(pose);

        kfusionCPU_.volume().setPositiveTsdfTruncDist(0.050f/*meters*/);
        kfusionCPU_.volume().setNegativeTsdfTruncDist(-0.030f/*meters*/);
    }

    ~kfusionCpuApp() { }

    sensor_msgs::ImagePtr map_to_image(const cv::Mat& map, std::string& encoding) {
        cv_bridge::CvImagePtr cv_image(new cv_bridge::CvImage);
        cv_image->image = map;
        cv_image->encoding = encoding;
        return cv_image->toImageMsg();
    }

    void process_image_callback(const sensor_msgs::ImageConstPtr& msg) {
        // convert the ROS image into an OpenCV image
        cv_bridge::CvImagePtr raw_depth_map = cv_bridge::toCvCopy(msg, msg->encoding);
        raw_depth_map->image.convertTo(raw_depth_map->image, CV_32FC1); raw_depth_map->image *= 0.001f; raw_depth_map->encoding = "32FC1";

        // run KinectFusion on the image.
        {
            //cvpr_tum::ScopeTime time(">>> KinectFusion Algorithm ...");
            kfusionCPU_(raw_depth_map->image);
        }

        {
            //cvpr_tum::ScopeTime time(">>> Publishing Data ...");
            publishTsdfVolume(kfusionCPU_.volume());
            //publishCameraTransformation(kfusionCPU_.getCameraPose().rotation(), kfusionCPU_.getCameraPose().translation() + kfusionCPU_.volume().getSize() / 2, "/camera_transformation_frame");
            publishCameraTransformation((Eigen::Affine3d)kfusionCPU_.getCameraPose(), "/camera_transformation_frame");
        }
    }

    void publishTsdfVolume(const cvpr_tum::TsdfVolume& vol) {
        kfusionCPU_msgs::Volume volume_msg;

        volume_msg.header.stamp = ros::Time::now();
        volume_msg.header.frame_id = "/tsdf_volume_frame";

        volume_msg.resolution.x = vol.getResolution().coeff(0);
        volume_msg.resolution.y = vol.getResolution().coeff(1);
        volume_msg.resolution.z = vol.getResolution().coeff(2);

        volume_msg.world_size.x = vol.getSize().coeff(0);
        volume_msg.world_size.y = vol.getSize().coeff(1);
        volume_msg.world_size.z = vol.getSize().coeff(2);

        volume_msg.voxel_size.x = vol.getVoxelSize().coeff(0);
        volume_msg.voxel_size.y = vol.getVoxelSize().coeff(1);
        volume_msg.voxel_size.z = vol.getVoxelSize().coeff(2);

        volume_msg.num_voxels = vol.getNumVoxels();

        volume_msg.pos_trunc_dist = vol.getPositiveTsdfTruncDist();
        volume_msg.neg_trunc_dist = vol.getNegativeTsdfTruncDist();

        volume_msg.data.resize(vol.getNumVoxels());
        std::copy(vol.begin(), vol.end(), volume_msg.data.begin());

        volume_pub_.publish(volume_msg);
    }

    void publishCameraTransformation(const Eigen::Affine3d& transform, const std::string& frame) {
        if(pose_pub_.getNumSubscribers() == 0) return;

        geometry_msgs::PoseWithCovarianceStampedPtr msg(new geometry_msgs::PoseWithCovarianceStamped);

        static int seq = 1;

        msg->header.seq = seq++;
        msg->header.frame_id = frame;
        msg->header.stamp = ros::Time::now();

        tf::Transform tmp;

        tf::TransformEigenToTF(transform, tmp);
        tf::poseTFToMsg(tmp, msg->pose.pose);

        msg->pose.covariance.assign(0.0);

        pose_pub_.publish(msg);
    }

    void reset_kfusion(const std_msgs::BoolConstPtr& reset_required) {
        if (reset_required->data)
            kfusionCPU_.reset();
    }

    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;
    ros::Subscriber depth_image_sub_;
    ros::Subscriber reset_sub_;
    ros::Publisher volume_pub_;
    ros::Publisher pose_pub_;

    double world_size_;

    cvpr_tum::kfusionCPU kfusionCPU_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "kinectfusion_app");

    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    kfusionCpuApp app(nh, nh_private);

    ROS_INFO("kfusion algorithm started...");

    ros::spin();

    return 0;
}
