#include <ros/ros.h> // for ros::init etc.
#include <cv_bridge/cv_bridge.h> // for CvImages
#include <image_transport/image_transport.h> // for sensor_msgs::Image
#include <sensor_msgs/image_encodings.h>
#include "kfusionCPU/kfusionCPU.h" // for kinect_fusion
#include <pcl17/visualization/pcl_visualizer.h>
#include <pcl17/visualization/image_viewer.h>
#include <pcl17/ros/conversions.h>
#include "marching_cubes.h"

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

    pcl17::visualization::ImageViewer viewerDepth_;

    kfusionCPU::DepthMap generated_depth_;
};

struct SceneCloudView
{
    SceneCloudView(float vsz, int viz) :
        viz_(viz), cube_added_(false), sphere_count_(0)
    {
        cloud_ptr_ = pcl17::PointCloud<pcl17::PointXYZ>::Ptr(new pcl17::PointCloud<pcl17::PointXYZ>);
        spheres_cloud_ptr_ = pcl17::PointCloud<pcl17::PointXYZ>::Ptr(new pcl17::PointCloud<pcl17::PointXYZ>);
        if (viz_)
        {
            cloud_viewer_
                    = pcl17::visualization::PCLVisualizer::Ptr(
                                                             new pcl17::visualization::PCLVisualizer("Scene Cloud Viewer"));

            cloud_viewer_->setBackgroundColor(0, 0, 0);
            cloud_viewer_->addCoordinateSystem(1.0);
            cloud_viewer_->initCameraParameters();
            cloud_viewer_->setPosition(20, 50);
            cloud_viewer_->setSize(1080, 720);
            //cloud_viewer_->camera_.clip[0] = 0.01;
            //cloud_viewer_->camera_.clip[1] = 10.01;

        }
    }

    void show(kfusionCPU& kfusion)
    {
        if (!viz_)
            return;

        TsdfVolume vol = kfusion.volume();

        cloud_ptr_->points.clear();
        //for(auto i = vmap.origin(); i < (vmap.origin() + vmap.num_elements()); ++i)
        //        cloud_ptr_->points.push_back( pcl17::PointXYZ((*i)(0), (*i)(1), (*i)(2)) );

        float voxel_size = vol.getVoxelSize()(0);
        for (int x = 0; x < 128; ++x)
        {
            for (int y = 0; y < 128; ++y)
            {
                for (int z = 0; z < 128; ++z)
                {
                    if (fabs(vol.v(x, y, z)) < 0.02)
                    {
                        cloud_ptr_->points.push_back(pcl17::PointXYZ(x*voxel_size, y*voxel_size, z*voxel_size));
                    }
                }
            }
        }
        //cloud_ptr_->width = (int)cloud_ptr_->points.size();
        //cloud_ptr_->height = 1;
        cloud_viewer_->removeAllPointClouds();
        cloud_viewer_->addPointCloud<pcl17::PointXYZ> (cloud_ptr_);
        cloud_viewer_->spinOnce();
    }

    void clearClouds(bool print_message = false)
    {
        cloud_viewer_->removeAllPointClouds();
        cloud_ptr_->points.clear();
        spheres_cloud_ptr_->points.clear();
        if (print_message)
            cout << "Clouds/Meshes were cleared" << endl;
    }

    void setViewerPose(const Eigen::Affine3f& viewer_pose)
    {
        Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f(0, 0, 0);
        Eigen::Vector3f look_at_vector = viewer_pose.rotation() * Eigen::Vector3f(0, 0, 1) + pos_vector;
        Eigen::Vector3f up_vector = viewer_pose.rotation() * Eigen::Vector3f(0, -1, 0);
    }

    void
    toggleCube(const Eigen::Vector3f& size)
    {
        if (!viz_)
            return;

        if (cube_added_)
            cloud_viewer_->removeShape("cube");
        else
          cloud_viewer_->addCube(size*0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

        cube_added_ = !cube_added_;
    }

    void showMesh(kfusionCPU& kfusion)
    {
        if (!viz_)
            return;

        //ScopeTimeT time ("Mesh Extraction");
        //cout << "\nGetting mesh... " << flush;

        if (!marching_cubes_)
            marching_cubes_ = CUDAMarchingCubes::Ptr(new CUDAMarchingCubes());

        TsdfVolume vol = kfusion.volume();

        unsigned int maxverts;
        float* vertOut;
        if (marching_cubes_->computeIsosurface(vol.begin(), vol.getResolution()(0), vol.getSize()(0), vertOut,
                                               &maxverts))
            cout << "marching cubes computed!" << endl;

        if (!vertOut) return;

        mesh_ptr_ = convertToMesh(vertOut, maxverts);

        free(vertOut);
        cloud_viewer_->removeAllPointClouds();

        Eigen::Vector3f sphere_location_ = kfusion.getCameraPose().translation() + kfusion.volume().getSize()/2;
        spheres_cloud_ptr_->points.push_back( pcl17::PointXYZ(sphere_location_.x(), sphere_location_.y(), sphere_location_.z()) );
        //std::string sp = "sp"+boost::lexical_cast<std::string>(sphere_count_);

        cloud_viewer_->addPointCloud<pcl17::PointXYZ>(spheres_cloud_ptr_, "spheres");//addSphere(p, 0.05/*radius*/, 1.f, 0.f, 0.f, sp);
        cloud_viewer_->setPointCloudRenderingProperties (pcl17::visualization::PCL17_VISUALIZER_POINT_SIZE, 2, "spheres");
        cloud_viewer_->setPointCloudRenderingProperties (pcl17::visualization::PCL17_VISUALIZER_COLOR, 1.0,0.0,0.0, "spheres");

        //sphere_count_++;

        if (mesh_ptr_)
            cloud_viewer_->addPolygonMesh(*mesh_ptr_);
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
    int sphere_count_;
    pcl17::PointCloud<pcl17::PointXYZ>::Ptr spheres_cloud_ptr_;
    CUDAMarchingCubes::Ptr marching_cubes_;
    pcl17::visualization::PCLVisualizer::Ptr cloud_viewer_;
    pcl17::PointCloud<pcl17::PointXYZ>::Ptr cloud_ptr_;
    boost::shared_ptr<pcl17::PolygonMesh> mesh_ptr_;
};

class kfusionCpuApp
{

public:

    kfusionCpuApp(float vsz, int viz) :
        scene_cloud_view_(vsz, viz), scan_mesh_(true)
    {
        // setup publishers
        std::string raw_depth_topic = nh_.resolveName("/kfusionCPU/depth/image");
        std::string filtered_depth_topic = nh_.resolveName("/kfusionCPU/depth/image_filtered");
        uint32_t queue_size = 1;
        raw_depth_publisher_ = nh_.advertise<sensor_msgs::Image> (raw_depth_topic, queue_size);
        filtered_depth_publisher_ = nh_.advertise<sensor_msgs::Image> (filtered_depth_topic, queue_size);

        // setup subscribers
        sub_ = nh_.subscribe("/camera/depth/image", 3, &kfusionCpuApp::process_image_callback, this);

        scene_cloud_view_.cloud_viewer_->registerKeyboardCallback (keyboard_callback, (void*)this);


        //Init Kfusion
        //Eigen::Vector3f volume_size = Eigen::Vector3f::Constant(vsz/*meters*/);
        //kfusionCPU_.volume().setSize(volume_size);

        //Eigen::Matrix3f R = Eigen::Matrix3f::Identity(); // * AngleAxisf( pcl17::deg2rad(-30.f), Vector3f::UnitX());
        //Eigen::Vector3f t = volume_size * 0.5f - Eigen::Vector3f(0, 0, volume_size(2) / 2 * 1.2f);

        //Eigen::Affine3f pose = Eigen::Translation3f(t) * Eigen::AngleAxisf(R);

        //kfusionCPU_.setInitalCameraPose(pose);
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
        //raw_depth_map->image.convertTo(raw_depth_map->image, CV_32FC1);
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

        //image_view_.showGeneratedDepth(kfusionCPU_);

        //scene_cloud_view_.show(kfusionCPU_);
        if (scan_mesh_)
        {
            //scan_mesh_ = false;
            scene_cloud_view_.showMesh(kfusionCPU_);
        }
    }

    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;
    ros::Publisher raw_depth_publisher_;
    ros::Publisher filtered_depth_publisher_;

    kfusionCPU kfusionCPU_;

    bool scan_mesh_;

    //ImageView image_view_;
    SceneCloudView scene_cloud_view_;

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    static void
    keyboard_callback(const pcl17::visualization::KeyboardEvent &e, void *cookie)
    {
        kfusionCpuApp* app = reinterpret_cast<kfusionCpuApp*> (cookie);

        int key = e.getKeyCode ();

        if (e.keyUp ())
            switch (key)
            {
                case (int)'c': case (int)'C': app->kfusionCPU_.reset(); app->scene_cloud_view_.clearClouds(); break;
                case (int)'b': case (int)'B': app->scene_cloud_view_.toggleCube(app->kfusionCPU_.volume().getSize()); break;

                default:
                    break;
            }
    }
};

int main(int argc, char** argv)
{
    //ros::Rate r(10);
    ROS_INFO("Initialising...");
    ros::init(argc, argv, "kinectfusion_app");
    ROS_INFO("Ready!");
    float volume_size = 3.f;
    int visualization = 1;
    kfusionCpuApp app(volume_size, visualization);
    ros::spin();
    //r.sleep();
}

