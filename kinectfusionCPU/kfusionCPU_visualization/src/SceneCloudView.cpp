#include "kfusionCPU_visualization/SceneCloudView.h"

#include <pcl17/visualization/image_viewer.h>

pcl17::visualization::PCLVisualizer::Ptr SceneCloudView::cloud_viewer_ = pcl17::visualization::PCLVisualizer::Ptr(
            new pcl17::visualization::PCLVisualizer(
                "Scene Cloud Viewer"));

SceneCloudView::SceneCloudView()
    : cube_added_(false), cam_added_(false), cur_view_added_(false), first_time_(true)
{
    spheres_cloud_ptr_ = pcl17::PointCloud<pcl17::PointXYZ>::Ptr(new pcl17::PointCloud<pcl17::PointXYZ>);
    current_cloud_ptr_ = pcl17::PointCloud<pcl17::PointXYZ>::Ptr(new pcl17::PointCloud<pcl17::PointXYZ>);

    //cloud_viewer_ = pcl17::visualization::PCLVisualizer::Ptr(new pcl17::visualization::PCLVisualizer("Scene Cloud Viewer"));

    cloud_viewer_->setBackgroundColor(0, 0, 0);
    cloud_viewer_->addCoordinateSystem(1.0);
    cloud_viewer_->initCameraParameters();

    //cloud_viewer_->setCameraPosition((double)pos_vector[0], (double)pos_vector[1], (double)pos_vector[2], (double)look_at_vector[0],
    //                                   (double)look_at_vector[1], (double)look_at_vector[2], (double)up_vector[0], (double)up_vector[1],
    //                                   (double)up_vector[2]);
    //cloud_viewer_->setCameraParameters(cam);
    cloud_viewer_->setPosition(100, 100);
    cloud_viewer_->setSize(1280, 960);

    //cloud_viewer_->createViewPort (0.0, 0, 0.5, 1.0, vp_1);
    //cloud_viewer_->createViewPort (0.5, 0, 1.0, 1.0, vp_2);
}

void SceneCloudView::reset(bool print_message)
{
    clearClouds(print_message);
    trajectory_visualizers_.clear();
}

void SceneCloudView::clearClouds(bool print_message)
{
    cloud_viewer_->removeAllPointClouds();
    spheres_cloud_ptr_->points.clear();
    current_cloud_ptr_->points.clear();
    if (print_message)
        cout << "Clouds/Meshes were cleared" << endl;
}

void SceneCloudView::setViewerPose(const Eigen::Affine3d& viewer_pose)
{
    Eigen::Vector3d pos_vector = viewer_pose * Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d look_at_vector = viewer_pose.rotation() * Eigen::Vector3d(0, 0, 0) + pos_vector;
    Eigen::Vector3d up_vector = viewer_pose.rotation() * Eigen::Vector3d(0, -1, 0);
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
    cloud_viewer_->updateCamera();
}

void SceneCloudView::toggleCube(const Eigen::Vector3f& size)
{
    if (cube_added_)
        cloud_viewer_->removeShape("cube");
    else
        cloud_viewer_->addCube(size * 0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

    cube_added_ = !cube_added_;
}

void SceneCloudView::toggleCamPose()
{
    cam_added_ = !cam_added_;
}

void SceneCloudView::toggleCurView()
{
    cur_view_added_ = !cur_view_added_;
}

void SceneCloudView::updateCellsCloud(const cvpr_tum::TsdfVolume::TsdfVolumeConstPtr& volume)
{
    current_cloud_ptr_->points.clear();

    float vx_sz = volume->getVoxelSize()(0);
    int v_res = volume->getResolution()(0);
    for (unsigned int x = 0; x < v_res; ++x)
        for (unsigned int y = 0; y < v_res; ++y)
            for (unsigned int z = 0; z < v_res; ++z)
                if (fabs(volume->v(z,y,x)) < 0.01f)
                    current_cloud_ptr_->points.push_back(pcl17::PointXYZ(x*vx_sz, y*vx_sz, z*vx_sz));
}

void SceneCloudView::update_current_cloud(const sensor_msgs::PointCloud2ConstPtr& msg, const Eigen::Affine3d& cam_pose, const Eigen::Vector3d& world_size)
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
        Eigen::Vector3d p(pt.x, pt.y, pt.z);
        Eigen::Vector3d g_p = cam_pose * p;// + world_size / 2;
        if ((g_p.array() < world_size.array()).all())
        {
            current_cloud_ptr_->points.push_back(pcl17::PointXYZ(g_p(0), g_p(1), g_p(2)));
        }
    }
}

void SceneCloudView::updateCamPCloud(const Eigen::Vector3d& cam_pose_t)
{
    spheres_cloud_ptr_->points.push_back(pcl17::PointXYZ(cam_pose_t.x(), cam_pose_t.y(),
                                                         cam_pose_t.z()));
}

void SceneCloudView::updateMesh(const cvpr_tum::TsdfVolume::TsdfVolumeConstPtr& volume)
{
    if (!marching_cubes_)
        marching_cubes_ = CUDAMarchingCubes::Ptr(new CUDAMarchingCubes());

    unsigned int maxverts;
    float* vertOut;
    if (marching_cubes_->computeIsosurface((float *)volume->data(),
                                           volume->getResolution()(0), volume->getSize()(0),
                                           vertOut, maxverts))
        cout << "marching cubes computed!" << endl;

    if (!vertOut)
        return;

    mesh_ptr_ = convertToMesh(vertOut, maxverts);

    free(vertOut);

}

void SceneCloudView::renderScene(const cvpr_tum::TsdfVolume::TsdfVolumeConstPtr& volume, bool has_gpu)
{
    cloud_viewer_->removeAllPointClouds();

    if (has_gpu)
    {
        updateMesh(volume);
        cloud_viewer_->addPolygonMesh(*mesh_ptr_);
    } else
    {
        updateCellsCloud(volume);
        cloud_viewer_->addPointCloud<pcl17::PointXYZ> (current_cloud_ptr_, "current_frame_point_cloud");
    }

    if (cam_added_)
    {
        cloud_viewer_->addPointCloud<pcl17::PointXYZ> (spheres_cloud_ptr_, "cam_poses");//addSphere(p, 0.05/*radius*/, 1.f, 0.f, 0.f, sp);
        cloud_viewer_->setPointCloudRenderingProperties(pcl17::visualization::PCL17_VISUALIZER_POINT_SIZE, 5, "cam_poses");
        cloud_viewer_->setPointCloudRenderingProperties(pcl17::visualization::PCL17_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "cam_poses");
    }

    if (cur_view_added_)
    {
        //pcl17::visualization::PointCloudColorHandlerCustom<pcl17::PointXYZ> tgt_h (current_cloud_ptr_, 0, 255, 0);
        //cloud_viewer_->addPointCloud<pcl17::PointXYZ> (current_cloud_ptr_, tgt_h, "current_frame_point_cloud", vp_1);
        cloud_viewer_->addPointCloud<pcl17::PointXYZ> (current_cloud_ptr_, "current_frame_point_cloud");
        cloud_viewer_->setPointCloudRenderingProperties(pcl17::visualization::PCL17_VISUALIZER_POINT_SIZE, 5, "current_frame_point_cloud");
        cloud_viewer_->setPointCloudRenderingProperties(pcl17::visualization::PCL17_VISUALIZER_COLOR, 0.0, 0.0, 1.0,
                                                        "current_frame_point_cloud");
    }

    for(TrajectoryVisualizerMap::iterator it = trajectory_visualizers_.begin(); it != trajectory_visualizers_.end(); ++it)
    {
        it->second->updateVisualizer(cloud_viewer_);
    }

    cloud_viewer_->spinOnce();
}

TrajectoryVisualizer::Ptr SceneCloudView::trajectory(std::string name)
{
    TrajectoryVisualizerMap::iterator trajectory = trajectory_visualizers_.find(name);

    if(trajectory_visualizers_.end() == trajectory)
    {
        trajectory = trajectory_visualizers_.insert(
                    std::make_pair(name, PclTrajectoryVisualizerPtr(new PclTrajectoryVisualizer(name)))
                    ).first;
    }

    return trajectory->second;
}

boost::shared_ptr<pcl17::PolygonMesh> SceneCloudView::convertToMesh(float* triangles, unsigned int maxverts)
{
    if (maxverts == 0)
        return boost::shared_ptr<pcl17::PolygonMesh>();

    pcl17::PointCloud<pcl17::PointXYZ> cloud;
    cloud.width = (int)(maxverts);
    cloud.height = 1;

    cloud.points.clear();
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

void SceneCloudView::addCoordinateSystem(const Eigen::Affine3d& pose, bool keep_last)
{
    if(!keep_last && !first_time_)
        cloud_viewer_->removeCoordinateSystem();

    cloud_viewer_->addCoordinateSystem(0.3, pose.cast<float>());

    trajectory("estimate")->color(Color::red()).add(pose);

    first_time_ = false;
}

//	void SceneCloudView::bindSwitchToKey()//(Switch& s, std::string& key)
void SceneCloudView::registerKeyboardCallback(void(*callback)(const pcl17::visualization::KeyboardEvent &, void *), void *cookie)
{
    //		cloud_viewer_->registerKeyboardCallback(&SceneCloudView::onSwitchKeyPressed, new SwitchKeyBinding(s, key));
    cloud_viewer_->registerKeyboardCallback(callback, cookie);
}

