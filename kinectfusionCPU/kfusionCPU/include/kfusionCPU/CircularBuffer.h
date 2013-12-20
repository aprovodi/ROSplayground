#ifndef INCLUDE_KFUSIONCPU_CIRCULARBUFFER_H_
#define INCLUDE_KFUSIONCPU_CIRCULARBUFFER_H_

#include <Eigen/Geometry>

#include "kfusionCPU/tsdfVolume.h"

namespace cvpr_tum {

class CircularBuffer
{
public:
    CircularBuffer (const float distance_threshold,  const float cube_size = 3.f, const int nb_voxels_per_axis = 64)
    {
        tsdf_memory_start = 0; tsdf_memory_end = 0; tsdf_rolling_buff_origin = 0;
        weights_memory_start = 0; weights_memory_end = 0; weights_rolling_buff_origin = 0;
        distance_threshold_ = distance_threshold;
        volume_size(0) = cube_size;
        volume_size(1) = cube_size;
        volume_size(2) = cube_size;
        volume_resolution(0) = nb_voxels_per_axis;
        volume_resolution(1) = nb_voxels_per_axis;
        volume_resolution(2) = nb_voxels_per_axis;
        origin_GRID = Eigen::Vector3i::Zero();
        origin_GRID_global = Eigen::Vector3i::Zero();
        origin_metric = Eigen::Vector3f::Zero();
    }

    bool checkForShift (const TsdfVolume::TsdfVolumePtr volume, const Eigen::Affine3f &cam_pose, const float distance_camera_target, const bool perform_shift = true);

    void performShift (const TsdfVolume::TsdfVolumePtr volume, const Eigen::Vector3f &target_point);

    void computeAndSetNewCubeMetricOrigin (const Eigen::Vector3f &target_point, int &shiftX, int &shiftY, int &shiftZ);

    /** \brief updates cyclical buffer origins given offsets on X, Y and Z
    * \param[in] tsdf_volume pointer to the TSDF volume managed by this cyclical buffer
    * \param[in] offset in indices
    */
    void shiftOrigin (TsdfVolume::TsdfVolumePtr tsdf_volume, const Eigen::Vector3i offset)
    {
        // shift rolling origin (making sure they keep in [0 - NbVoxels[ )
        origin_GRID(0) += offset(0);
        if(origin_GRID(0) >= volume_resolution(0))
            origin_GRID(0) -= volume_resolution(0);
        else if(origin_GRID(0) < 0)
            origin_GRID(0) += volume_resolution(0);

        origin_GRID(1) += offset(1);
        if(origin_GRID(1) >= volume_resolution(1))
            origin_GRID(1) -= volume_resolution(1);
        else if(origin_GRID(1) < 0)
            origin_GRID(1) += volume_resolution(1);

        origin_GRID(2) += offset(2);
        if(origin_GRID(2) >= volume_resolution(2))
            origin_GRID(2) -= volume_resolution(2);
        else if(origin_GRID(2) < 0)
            origin_GRID(2) += volume_resolution(2);

        std::cout << "origin_GRID: " << origin_GRID(0) << " " << origin_GRID(1) << " " << origin_GRID(2) << std::endl;

        // update memory pointers
        tsdf_memory_start = tsdf_volume->begin();
        tsdf_memory_end = tsdf_volume->end();
        tsdf_rolling_buff_origin = tsdf_volume->begin() + (int)(volume_resolution(0) * (volume_resolution(1) * origin_GRID(2) + (origin_GRID(1)) + origin_GRID(0)));
        weights_memory_start = tsdf_volume->begin_weights();
        weights_memory_end = tsdf_volume->end_weights();
        weights_rolling_buff_origin = tsdf_volume->begin_weights() + (int)(volume_resolution(0) * (volume_resolution(1) * (origin_GRID(2)) + (origin_GRID(1)) + origin_GRID(0)));

        // update global origin
        origin_GRID_global += offset;
        std::cout << "origin_GRID_global: " << origin_GRID_global(0) << " " << origin_GRID_global(1) << " " << origin_GRID_global(2) << std::endl;
    }

    void clearTSDFSlice( TsdfVolume::TsdfVolumePtr tsdf_volume, int shiftX, int shiftY, int shiftZ );

    void get_shift_value()
    {
        std::cout << "pointer shift: " << tsdf_rolling_buff_origin - tsdf_memory_start << std::endl;
    }

    void shift_tsdf_pointer(float** value, float** weight)
    {
        ///Shift the pointer by (@origin - @start)
        //int s = (tsdf_rolling_buff_origin - tsdf_memory_start);
        //std::cout << "1D shift: " << s << std::endl;
        *value += (tsdf_rolling_buff_origin - tsdf_memory_start);
        *weight += (weights_rolling_buff_origin - weights_memory_start);

        ///If we land outside of the memory, make sure to "modulo" the new value
        if(*value > tsdf_memory_end)
        {
            //std::cout << "1D shift: modulo" << std::endl;
            *value -= (tsdf_memory_end - tsdf_memory_start + 1);
            *weight -= (weights_memory_end - weights_memory_start + 1);
        }
    }

    void shift_tsdf_pointer(float** value)
    {
        *value += (tsdf_rolling_buff_origin - tsdf_memory_start);
        if(*value > tsdf_memory_end)
            *value -= (tsdf_memory_end - tsdf_memory_start + 1);
    }

    Eigen::Vector3i shift_tsdf_indeces(int i, int j, int k)
    {
        return Eigen::Vector3i((i+origin_GRID(0))%volume_resolution(0),
                                (j+origin_GRID(1))%volume_resolution(1),
                                (k+origin_GRID(2))%volume_resolution(2));
    }

    /** \brief Reset buffer structure
    * \param[in] tsdf_volume pointer to the TSDF volume managed by this cyclical buffer
    */
    void resetBuffer (TsdfVolume::TsdfVolumePtr tsdf_volume)
    {
        origin_GRID = Eigen::Vector3i::Zero();
        origin_GRID_global = Eigen::Vector3i::Zero();
        origin_metric = Eigen::Vector3f::Zero();
        tsdf_memory_start = tsdf_volume->begin();
        tsdf_memory_end = tsdf_volume->end();
        tsdf_rolling_buff_origin = tsdf_memory_start;
        //shiftOrigin (tsdf_volume, Eigen::Vector3i(-30, 0, 0));
    }

    /* Sets the distance threshold between cube's center and target point that triggers a shift. */
    void setDistanceThreshold (const float threshold) { distance_threshold_ = threshold; }

    /* Set the physical size represented by the default TSDF volume. */
    void setVolumeSize(const Eigen::Vector3f& size) { volume_size = size; }

    inline const float* tsdf_start() { return tsdf_memory_start; }

    inline const float* tsdf_end() { return tsdf_memory_end; }

    inline const Eigen::Vector3f& get_current_origin () { return origin_metric; }

    inline const Eigen::Vector3i& get_origin () { return origin_GRID; }
    inline const Eigen::Vector3i& get_global_origin () { return origin_GRID_global; }

    void computeMinMaxBounds(int newX, int newY, int newZ);

private:
    float distance_threshold_;

    /** \brief Address of the first element of the TSDF volume in memory*/
    float* tsdf_memory_start;
    float* weights_memory_start;
    /** \brief Address of the last element of the TSDF volume in memory*/
    float* tsdf_memory_end;
    float* weights_memory_end;
    /** \brief Memory address of the origin of the rolling buffer. MUST BE UPDATED AFTER EACH SHIFT.*/
    float* tsdf_rolling_buff_origin;
    float* weights_rolling_buff_origin;

    /** \brief Internal cube origin for rollign buffer.*/
    Eigen::Vector3i origin_GRID;
    /** \brief Cube origin in world coordinates.*/
    Eigen::Vector3i origin_GRID_global;
    /** \brief Current metric origin of the cube, in world coordinates.*/
    Eigen::Vector3f origin_metric;

    Eigen::Vector3f volume_size;
    Eigen::Vector3i volume_resolution; //64

    /** new shifting boundaries **/
    Eigen::Vector3i minBounds_, maxBounds_;

};

}  // namespace cvpr_tum
#endif  // INCLUDE_KFUSIONCPU_CIRCULARBUFFER_H_
