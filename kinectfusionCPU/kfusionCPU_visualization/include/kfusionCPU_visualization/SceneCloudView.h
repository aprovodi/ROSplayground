#ifndef SCENE_VIEW_VISUALIZER_H_
#define SCENE_VIEW_VISUALIZER_H_

#include <map>
#include <boost/shared_ptr.hpp>

#include <Eigen/Geometry>
#include <pcl17/point_types.h>
#include <pcl17/visualization/pcl_visualizer.h>

#include "kfusionCPU_visualization/pcl_trajectory_visualizer.h"

#include "kfusionCPU/tsdfVolume.h"
#include "marching_cubes.h"

class SceneCloudView
{
public:
	typedef boost::shared_ptr<PclTrajectoryVisualizer> PclTrajectoryVisualizerPtr;
	typedef std::map<std::string, PclTrajectoryVisualizerPtr> TrajectoryVisualizerMap;  

	SceneCloudView();

	void reset(bool print_message = false);

    void clearClouds(bool print_message = false);

    void setViewerPose(const Eigen::Affine3d& viewer_pose);

    void toggleCube(const Eigen::Vector3f& size);

    void toggleCamPose();

    void toggleCurView();

    void updateCellsCloud(const cvpr_tum::TsdfVolume::TsdfVolumeConstPtr& volume);

    void update_current_cloud(const sensor_msgs::PointCloud2ConstPtr& msg, const Eigen::Affine3d& cam_pose, const Eigen::Vector3d& world_size);

    void updateCamPCloud(const Eigen::Vector3d& cam_pose_t);

    void updateMesh(const cvpr_tum::TsdfVolume::TsdfVolumeConstPtr& volume);

    void renderScene(const cvpr_tum::TsdfVolume::TsdfVolumeConstPtr& volume, bool has_gpu = true);

    void addCoordinateSystem(const Eigen::Affine3d& pose, bool keep_last = true);

	static void registerKeyboardCallback(void(*callback)(const pcl17::visualization::KeyboardEvent &, void *), void *cookie=NULL);

private:

    boost::shared_ptr<pcl17::PolygonMesh> convertToMesh(float* triangles, unsigned int maxverts);

	TrajectoryVisualizer::Ptr trajectory(std::string name);

	TrajectoryVisualizerMap trajectory_visualizers_;
    int viz_;
    bool cube_added_;
    bool cam_added_;
    bool cur_view_added_;
    CUDAMarchingCubes::Ptr marching_cubes_;
    static pcl17::visualization::PCLVisualizer::Ptr cloud_viewer_;
    pcl17::PointCloud<pcl17::PointXYZ>::Ptr current_cloud_ptr_;
    pcl17::PointCloud<pcl17::PointXYZ>::Ptr spheres_cloud_ptr_;
    boost::shared_ptr<pcl17::PolygonMesh> mesh_ptr_;
    bool first_time_;
    int vp_1;
    int vp_2;

	//void SceneCloudView::bindSwitchToKey(Switch& s, std::string& key);
	//static void SceneCloudView::onSwitchKeyPressed(const pcl17::visualization::KeyboardEvent& e, void* data);

	struct SwitchKeyBinding
	{
	    Switch& s;
	    std::string key;
	    SwitchKeyBinding(Switch& s, std::string& key) : s(s), key(key) { }
	};

};
#endif /* SCENE_VIEW_VISUALIZER_H_ */
