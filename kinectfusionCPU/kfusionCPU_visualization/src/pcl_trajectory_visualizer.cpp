#include "kfusionCPU_visualization/pcl_trajectory_visualizer.h"

#include <pcl17/visualization/common/common.h>

PclTrajectoryVisualizer::PclTrajectoryVisualizer(std::string& name) { name_ = name; }

TrajectoryVisualizer& PclTrajectoryVisualizer::add(const Eigen::Affine3d& pose)
{
    PointPair latest;
    latest.second.x = pose.translation()(0);
    latest.second.y = pose.translation()(1);
    latest.second.z = pose.translation()(2);

    if(pairs_.empty())
    {
      pairs_.push_back(latest); // add to list - this copies the object
      pairs_.back().first = &(pairs_.back().second); // set pointer to second point in newly allocated object
      last_ = pairs_.begin();
    }

    latest.first = &(pairs_.back().second);

    {
      pairs_.push_back(latest);
    }

    return *this;
  }

  void PclTrajectoryVisualizer::updateVisualizer(pcl17::visualization::PCLVisualizer::Ptr& visualizer)
  {
    if(pairs_.empty()) return;

    pcl17::PointCloud<pcl17::PointXYZ>::Ptr cloud(new pcl17::PointCloud<pcl17::PointXYZ>);

    for(list::iterator it = pairs_.begin(); it != pairs_.end(); ++it)
    {
      cloud->points.push_back(it->second);
    }
    for(list::reverse_iterator it = pairs_.rbegin(); it != pairs_.rend(); ++it)
    {
      cloud->points.push_back(it->second);
    }

    visualizer->removeShape(name());
    visualizer->addPolygon<pcl17::PointXYZ>(cloud, name());
    visualizer->setShapeRenderingProperties(pcl17::visualization::PCL17_VISUALIZER_COLOR,  color().r, color().g, color().b, name());
    visualizer->setShapeRenderingProperties(pcl17::visualization::PCL17_VISUALIZER_LINE_WIDTH, 4, name());
  }
