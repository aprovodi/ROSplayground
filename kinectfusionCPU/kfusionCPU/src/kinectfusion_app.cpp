#include <ros/ros.h> // for ros::init etc.
#include <cv_bridge/cv_bridge.h> // for CvImages
#include <image_transport/image_transport.h> // for sensor_msgs::Image
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Bool.h>
#include "kfusionCPU/kfusionCPU.h" // for kinect_fusion
#include "kfusionCPU_msgs/Volume.h"
#include "kfusionCPU_msgs/Transformation.h"

using namespace cvpr_tum;

struct kfusionCpuApp
{
    kfusionCpuApp(float vsz)
    {
        // setup subscribers
        depth_image_sub_ = nh_.subscribe("/camera/depth/image", 1, &kfusionCpuApp::process_image_callback, this);
        reset_sub_ = nh_.subscribe("/reset", 1, &kfusionCpuApp::reset_kfusion, this);

        // setup publishers
        uint32_t queue_size = 1;

        std::string topic_volume = nh_.resolveName("/kfusionCPU/tsdf_volume");
        volume_publisher_ = nh_.advertise<kfusionCPU_msgs::Volume> (topic_volume, queue_size);

        std::string topic_translation = nh_.resolveName("/kfusionCPU/transformations");
        cam_transform_publisher_ = nh_.advertise<kfusionCPU_msgs::Transformation> (topic_translation, queue_size);

        // init Kfusion
        Eigen::Vector3f volume_size = Eigen::Vector3f::Constant(vsz/*meters*/);
        kfusionCPU_.volume().setSize(volume_size);

        //Eigen::Matrix3f R = Eigen::Matrix3f::Identity(); // * AngleAxisf( pcl17::deg2rad(-30.f), Vector3f::UnitX());
        //Eigen::Vector3f t = volume_size * 0.5f - Eigen::Vector3f(0, 0, volume_size(2) / 2 * 1.2f);

        //Eigen::Affine3f pose = Eigen::Translation3f(t) * Eigen::AngleAxisf(R);

        //kfusionCPU_.setInitalCameraPose(pose);
    }

    ~kfusionCpuApp()
    {
    }

    sensor_msgs::ImagePtr map_to_image(const cv::Mat& map, std::string& encoding)
    {
        cv_bridge::CvImagePtr cv_image(new cv_bridge::CvImage);
        cv_image->image = map;
        cv_image->encoding = encoding;
        return cv_image->toImageMsg();
    }

    void process_image_callback(const sensor_msgs::ImageConstPtr& msg)
    {
        // convert the ROS image into an OpenCV image
        cv_bridge::CvImagePtr raw_depth_map = cv_bridge::toCvCopy(msg, msg->encoding);
        //raw_depth_map->image.convertTo(raw_depth_map->image, CV_32FC1); raw_depth_map->image *= 0.001f; raw_depth_map->encoding = "32FC1";

        // run KinectFusion on the image.
        {
            ScopeTime time(">>> KinectFusion Algorithm ...");
            kfusionCPU_(raw_depth_map->image);
        }

        {
            ScopeTime time(">>> Publishing Data ...");
            if (volume_publisher_.getNumSubscribers() > 0)
                publishTsdfVolume(kfusionCPU_.volume());

            if (cam_transform_publisher_.getNumSubscribers() > 0)
                publishCameraTransformation(kfusionCPU_.getCameraPose().rotation(), kfusionCPU_.getCameraPose().translation() + kfusionCPU_.volume().getSize() / 2);
        }
    }

    void publishTsdfVolume(const TsdfVolume& vol)
    {
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

        volume_publisher_.publish(volume_msg);
    }
    void publishCameraTransformation(const Eigen::Matrix3f& Rcam, const Eigen::Vector3f& tcam)
    {
        kfusionCPU_msgs::Transformation transformation_msg;

        transformation_msg.header.stamp = ros::Time::now();
        transformation_msg.header.frame_id = "/camera_transformation_frame";

        std::copy(Rcam.data(), Rcam.data() + 9, transformation_msg.rotation.begin());

        transformation_msg.translation.x = tcam.coeff(0);
        transformation_msg.translation.y = tcam.coeff(1);
        transformation_msg.translation.z = tcam.coeff(2);

        cam_transform_publisher_.publish(transformation_msg);
    }

    void reset_kfusion(const std_msgs::BoolConstPtr& reset_required)
    {
        if (reset_required->data)
            kfusionCPU_.reset();
    }

    ros::NodeHandle nh_;
    ros::Subscriber depth_image_sub_;
    ros::Subscriber reset_sub_;
    ros::Publisher volume_publisher_;
    ros::Publisher cam_transform_publisher_;

    kfusionCPU kfusionCPU_;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "kinectfusion_app");
    float volume_size = 3.f;
    kfusionCpuApp app(volume_size);
    ros::spin();
}
