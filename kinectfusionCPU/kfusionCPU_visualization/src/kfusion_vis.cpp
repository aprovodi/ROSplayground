#include <ros/ros.h> // for ros::init etc.
#include <cv_bridge/cv_bridge.h> // for CvImages
#include <pcl17/visualization/pcl_visualizer.h>
#include <pcl17/visualization/image_viewer.h>
#include <pcl17/ros/conversions.h>
#include <pcl17_ros/point_cloud.h>
#include <boost/foreach.hpp>
#include <std_msgs/Bool.h>
#include <sensor_msgs/PointCloud2.h>
#include "marching_cubes.h"
#include "kfusionCPU_msgs/Volume.h"
#include "kfusionCPU_msgs/Transformation.h"
#include "kfusionCPU/tsdfVolume.h"
#include "kfusionCPU/kfusionCPU.h"

using namespace cvpr_tum;

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

struct SceneCloudView
{
    SceneCloudView(float vsz) : cube_added_(false), cam_added_(false), cur_view_added_(false)
    {
        spheres_cloud_ptr_ = pcl17::PointCloud<pcl17::PointXYZ>::Ptr(new pcl17::PointCloud<pcl17::PointXYZ>);
        current_cloud_ptr_ = pcl17::PointCloud<pcl17::PointXYZ>::Ptr(new pcl17::PointCloud<pcl17::PointXYZ>);

            cloud_viewer_
                    = pcl17::visualization::PCLVisualizer::Ptr(
                                                               new pcl17::visualization::PCLVisualizer(
                                                                                                       "Scene Cloud Viewer"));

            cloud_viewer_->setBackgroundColor(0.3, 0.3, 0.3);
            cloud_viewer_->addCoordinateSystem(2.0);
            cloud_viewer_->initCameraParameters();
            cloud_viewer_->setPosition(100, 100);
            cloud_viewer_->setSize(1280, 960);

            //cloud_viewer_->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
            //cloud_viewer_->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
    }

    void clearClouds(bool print_message = false)
    {
        cloud_viewer_->removeAllPointClouds();
        spheres_cloud_ptr_->points.clear();
        current_cloud_ptr_->points.clear();
        if (print_message)
            cout << "Clouds/Meshes were cleared" << endl;
    }

    void setViewerPose(const Eigen::Affine3f& viewer_pose)
    {
        Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
        Eigen::Vector3f look_at_vector = viewer_pose.rotation() * Eigen::Vector3f(0, 0, 0) + pos_vector;
        Eigen::Vector3f up_vector = viewer_pose.rotation() * Eigen::Vector3f(0, -1, 0);
        pcl17::visualization::Camera cam;
        cam.pos[0] = pos_vector[0];
        cam.pos[1] = pos_vector[1];
        cam.pos[2] = pos_vector[2];
        cam.focal[0] = look_at_vector[0];
        cam.focal[1] = look_at_vector[1];
        cam.focal[2] = look_at_vector[2];
        cam.view[0] = up_vector[0];
        cam.view[1] = up_vector[1];
        cam.view[2] = up_vector[2];
        cloud_viewer_->setCameraPosition((double)pos_vector[0], (double)pos_vector[1], (double)pos_vector[2], (double)look_at_vector[0],
                                           (double)look_at_vector[1], (double)look_at_vector[2], (double)up_vector[0], (double)up_vector[1],
                                           (double)up_vector[2]);
        //cloud_viewer_->updateCamera();
    }

    void toggleCube(const Eigen::Vector3f& size)
    {
        if (cube_added_)
            cloud_viewer_->removeShape("cube");
        else
            cloud_viewer_->addCube(size * 0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

        cube_added_ = !cube_added_;
    }

    void toggleCamPose()
    {
        cam_added_ = !cam_added_;
    }

    void toggleCurView()
    {
        cur_view_added_ = !cur_view_added_;
    }

    void update_current_cloud(const sensor_msgs::PointCloud2ConstPtr& msg, const Eigen::Affine3f& cam_pose, const Eigen::Vector3f& world_size)
    {
        if (!cur_view_added_)
            return;
        pcl17::PointCloud<pcl17::PointXYZ> cloud;
        pcl17::fromROSMsg(*msg, cloud);

        current_cloud_ptr_->points.clear();
        current_cloud_ptr_->width = (int)cloud.points.size();
        current_cloud_ptr_->height = 1;

        BOOST_FOREACH (const pcl17::PointXYZ& pt, cloud.points)
        {
            Eigen::Vector3f p(pt.x, pt.y, pt.z);
            Eigen::Vector3f g_p = cam_pose * p;// + world_size / 2;
            if ((g_p.array() < world_size.array()).all())
            {
                current_cloud_ptr_->points.push_back(pcl17::PointXYZ(g_p(0), g_p(1), g_p(2)));
            }
        }
    }

    void updateCamPCloud(const Eigen::Vector3f& cam_pose_t)
    {
        spheres_cloud_ptr_->points.push_back(pcl17::PointXYZ(cam_pose_t.x(), cam_pose_t.y(),
                                                             cam_pose_t.z()));
    }

    void showMesh(const TsdfVolume::TsdfVolumeConstPtr& volume)
    {
        cloud_viewer_->removeAllPointClouds();

        if (!marching_cubes_)
            marching_cubes_ = CUDAMarchingCubes::Ptr(new CUDAMarchingCubes());

        unsigned int maxverts;
        float* vertOut;
        if (marching_cubes_->computeIsosurface((float *)volume->data(), volume->getResolution()(0),
                                               volume->getSize()(0), vertOut, &maxverts))
            cout << "marching cubes computed!" << endl;

        if (!vertOut)
            return;

        mesh_ptr_ = convertToMesh(vertOut, maxverts);

        free(vertOut);

        cloud_viewer_->addPolygonMesh(*mesh_ptr_);

        if (cam_added_)
        {
            cloud_viewer_->addPointCloud<pcl17::PointXYZ> (spheres_cloud_ptr_, "cam_poses");//addSphere(p, 0.05/*radius*/, 1.f, 0.f, 0.f, sp);
            cloud_viewer_->setPointCloudRenderingProperties(pcl17::visualization::PCL17_VISUALIZER_POINT_SIZE, 5, "cam_poses");
            cloud_viewer_->setPointCloudRenderingProperties(pcl17::visualization::PCL17_VISUALIZER_COLOR, 1.0, 0.0, 0.0,
                                                        "cam_poses");
        }

        if (cur_view_added_)
        {
            //pcl17::visualization::PointCloudColorHandlerCustom<pcl17::PointXYZ> tgt_h (current_cloud_ptr_, 0, 255, 0);
            //cloud_viewer_->addPointCloud<pcl17::PointXYZ> (current_cloud_ptr_, tgt_h, "current_frame_point_cloud", vp_1);
            cloud_viewer_->addPointCloud<pcl17::PointXYZ> (current_cloud_ptr_, "current_frame_point_cloud");
            cloud_viewer_->setPointCloudRenderingProperties(pcl17::visualization::PCL17_VISUALIZER_COLOR, 0.0, 0.0, 1.0,
                                                        "current_frame_point_cloud");
        }

        cloud_viewer_->spinOnce();
    }

    boost::shared_ptr<pcl17::PolygonMesh> convertToMesh(float* triangles, unsigned int maxverts)
    {
        if (maxverts == 0)
            return boost::shared_ptr<pcl17::PolygonMesh>();

        pcl17::PointCloud<pcl17::PointXYZ> cloud;
        cloud.points.clear();
        cloud.width = (int)(maxverts);
        cloud.height = 1;

        for (uint i = 0; i < 3 * maxverts; i += 3)
            cloud.points.push_back(pcl17::PointXYZ(triangles[i], triangles[i + 1], triangles[i + 2]));

        boost::shared_ptr<pcl17::PolygonMesh> mesh_ptr(new pcl17::PolygonMesh());
        try
        {
            pcl17::toROSMsg(cloud, mesh_ptr->cloud);
        }
        catch (std::runtime_error e)
        {
            ROS_ERROR_STREAM("Error in converting cloud to image message: "
                    << e.what());
        }

        mesh_ptr->polygons.resize(maxverts / 3);

        for (size_t i = 0; i < mesh_ptr->polygons.size(); ++i)
        {
            pcl17::Vertices v;
            v.vertices.push_back(i * 3 + 0);
            v.vertices.push_back(i * 3 + 2);
            v.vertices.push_back(i * 3 + 1);
            mesh_ptr->polygons[i] = v;
        }
        return mesh_ptr;
    }

    int viz_;
    bool cube_added_;
    bool cam_added_;
    bool cur_view_added_;
    CUDAMarchingCubes::Ptr marching_cubes_;
    pcl17::visualization::PCLVisualizer::Ptr cloud_viewer_;
    pcl17::PointCloud<pcl17::PointXYZ>::Ptr current_cloud_ptr_;
    pcl17::PointCloud<pcl17::PointXYZ>::Ptr spheres_cloud_ptr_;
    boost::shared_ptr<pcl17::PolygonMesh> mesh_ptr_;
    int vp_1;
    int vp_2;
};

struct kfusionCpuVis
{
    kfusionCpuVis(float vsz) : scene_cloud_view_(vsz),
            volume_resolution_(TsdfVolume::VOLUME_X, TsdfVolume::VOLUME_Y, TsdfVolume::VOLUME_Z)
    {
        tsdf_volume_ = TsdfVolume::TsdfVolumePtr(new TsdfVolume(volume_resolution_));

        // setup publishers
        std::string raw_depth_topic = nh_.resolveName("/kfusionCPU/depth/image");
        std::string filtered_depth_topic = nh_.resolveName("/kfusionCPU/depth/image_filtered");
        uint32_t queue_size = 1;
        raw_depth_publisher_ = nh_.advertise<sensor_msgs::Image> (raw_depth_topic, queue_size);
        filtered_depth_publisher_ = nh_.advertise<sensor_msgs::Image> (filtered_depth_topic, queue_size);
        reset_publisher_ = nh_.advertise<std_msgs::Bool> ("/reset", queue_size);

        // setup subscribers
        volume_sub_ = nh_.subscribe("/kfusionCPU/tsdf_volume", 1, &kfusionCpuVis::process_volume_callback, this);
        cam_translation_sub_ = nh_.subscribe("/kfusionCPU/transformations", 1, &kfusionCpuVis::process_cam_translation_callback, this);
        cur_pc_sub_ = nh_.subscribe("/camera/depth_registered/points", 1, &kfusionCpuVis::process_cur_pc_callback, this);

        scene_cloud_view_.cloud_viewer_->registerKeyboardCallback(keyboard_callback, (void*)this);
    }

    ~kfusionCpuVis()
    {
    }

    void process_volume_callback(const kfusionCPU_msgs::VolumeConstPtr& msg)
    {
        world_size_(0) = msg->world_size.x;
        world_size_(1) = msg->world_size.y;
        world_size_(2) = msg->world_size.z;

        tsdf_volume_->setSize(world_size_);
        tsdf_volume_->copyFrom((float *)msg->data.data(), msg->num_voxels);
        {
            ScopeTime time(">>> VISUALIZATION.......................");

            scene_cloud_view_.showMesh(tsdf_volume_);
        }
    }

    void process_cam_translation_callback(const kfusionCPU_msgs::TransformationConstPtr& msg)
    {
        Eigen::Matrix3f Rcam;
        std::copy(msg->rotation.begin(), msg->rotation.end(), Rcam.data());

        Eigen::Vector3f tcam(msg->translation.x, msg->translation.y, msg->translation.z);

        cam_pose_.linear() = Rcam;
        cam_pose_.translation() = tcam;

        //scene_cloud_view_.setViewerPose(cam_pose_);
        scene_cloud_view_.updateCamPCloud(tcam);
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
                    app->scene_cloud_view_.clearClouds();
                    break;
                }
                case (int)'b':
                case (int)'B':
                    app->scene_cloud_view_.toggleCube(app->world_size_);
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
    ros::Publisher pub_;
    ros::Subscriber volume_sub_;
    ros::Subscriber cam_translation_sub_;
    ros::Subscriber cur_pc_sub_;
    ros::Publisher raw_depth_publisher_;
    ros::Publisher filtered_depth_publisher_;
    ros::Publisher reset_publisher_;

    TsdfVolume::TsdfVolumePtr tsdf_volume_;

    const Eigen::Vector3i volume_resolution_; //number of voxels
    Eigen::Vector3f world_size_;
    Eigen::Affine3f cam_pose_;

    SceneCloudView scene_cloud_view_;
};

int main(int argc, char** argv)
{
    ROS_INFO("Initialising...");
    ros::init(argc, argv, "kinectfusion_vis");
    ROS_INFO("Ready!");
    float volume_size = 3.f;
    kfusionCpuVis app(volume_size);
    ros::spin();
}
