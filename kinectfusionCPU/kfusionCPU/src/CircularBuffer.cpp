#include "kfusionCPU/CircularBuffer.h"

bool cvpr_tum::CircularBuffer::checkForShift (const TsdfVolume::TsdfVolumePtr volume, const Eigen::Affine3f &cam_pose, const float distance_camera_target, const bool perform_shift)
{
    bool result = false;

    // project the target point in the cube
    Eigen::Vector3f targetPoint;
    targetPoint(0) = 0.0f;
    targetPoint(1) = 0.0f;
    targetPoint(2) = distance_camera_target; // place the point at camera position + distance_camera_target on Z
    targetPoint = cam_pose * targetPoint;
    targetPoint -= volume_size / 2;

    //std::cout << "cam_pose : " << cam_pose.translation()(0) << " " << cam_pose.translation()(1) << " " << cam_pose.translation()(2) << std::endl;
    std::cout << "targetPoint : " << targetPoint(0) << " " << targetPoint(1) << " " << targetPoint(2) << std::endl;

//    std::cout << "targetPoint: " << targetPoint << std::endl;

    // check distance from the cube's center
    Eigen::Vector3f center_cube;
    center_cube = origin_metric + volume_size / 2.0f;

//    std::cout << "origin_metric: " << origin_metric << std::endl;

//    std::cout << "euclideanDistance: " << (targetPoint - center_cube).norm() << std::endl;

    if ((targetPoint - origin_metric).norm() > distance_threshold_)
        result = true;

    if (!perform_shift)
        return (result);

    // perform shifting operations
    if (result)
        performShift (volume, targetPoint);

    return (result);
}

void cvpr_tum::CircularBuffer::performShift (const TsdfVolume::TsdfVolumePtr volume, const Eigen::Vector3f &target_point)
{
    // compute new origin and offsets
    int offset_X, offset_Y, offset_Z;
    computeAndSetNewCubeMetricOrigin (target_point, offset_X, offset_Y, offset_Z);

    int newX = origin_GRID(0) + offset_X;
    int newY = origin_GRID(0) + offset_Y;
    int newZ = origin_GRID(0) + offset_Z;

    computeMinMaxBounds(newX, newY, newZ);


    /*
    unsigned int idx_curr = 0;
    for (int x = 0; x < volume_resolution.x; ++x) {
        for (int y = 0; y < volume_resolution.y; ++y) {
            for (int z = 0; z < volume_resolution.z; ++z) {

                idx_curr = x * num_rows_ * num_slices_ + y * num_slices_ + z;
                unsigned int F = shifted_tsdf_pointer(idx_curr);

                // The black zone is the name given to the subvolume within the TSDF Volume grid that is shifted out.
                // In other words, the set of points in the TSDF grid that we want to extract in order to add it to the world model being built in CPU.
                bool in_black_zone = ( (x >= minBounds.x && x <= maxBounds.x) || (y >= minBounds.y && y <= maxBounds.y) || ( z >= minBounds.z && z <= maxBounds.z) ) ;
                float4 points[MAX_LOCAL_POINTS];
                int local_count = 0;

                if (in_black_zone)
                {
                    int W;
                    float F = fetch (buffer, x, y, z, W);

                    if (W != 0.0f && F != 1.f && F < 0.98 && F != 0.0f && F > -1.0f)
                    {
                        float4 p;
                        p.x = x;
                        p.y = y;
                        p.z = z;
                        p.w = F;
                        points[local_count++] = p;
                    }
                } // if (x < VOLUME_X && y < VOLUME_Y)
            }
        }
    }
    */
    /*
    // transform the slice from local to global coordinates
    Eigen::Affine3f global_cloud_transformation;
    global_cloud_transformation.translation ()[0] = origin_GRID_global.x;
    global_cloud_transformation.translation ()[1] = origin_GRID_global.y;
    global_cloud_transformation.translation ()[2] = origin_GRID_global.z;
    global_cloud_transformation.linear () = Eigen::Matrix3f::Identity ();
    pcl17::transformPointCloud (*current_slice, *current_slice, global_cloud_transformation);
    */

    // clear buffer slice and update the world model
    //clearTSDFSlice (volume, offset_X, offset_Y, offset_Z);

    // shift buffer addresses
    shiftOrigin (volume, Eigen::Vector3i(offset_X, offset_Y, offset_Z));

    /*
    // push existing data in the TSDF buffer
    if (previously_existing_slice->points.size () != 0 ) {
      volume->pushSlice(previously_existing_slice, getBuffer () );
    }
    */
}

void cvpr_tum::CircularBuffer::computeAndSetNewCubeMetricOrigin (const Eigen::Vector3f &target_point, int &shiftX, int &shiftY, int &shiftZ)
{
    // compute new origin for the cube, based on the target point
    Eigen::Vector3f new_cube_origin_meters;
    new_cube_origin_meters = target_point;// - volume_size / 2.0f;
    //new_cube_origin_meters(2) = target_point(2) - volume_size(2) / 2.0f;
    std::cout << "The old cube's metric origin was: " << origin_metric << std::endl;
    std::cout << "The new cube's metric origin is now: " << new_cube_origin_meters << std::endl;

    // deduce each shift in indices
    shiftX = (int)( (new_cube_origin_meters(0) - origin_metric(0)) * ( volume_resolution(0) / (float) (volume_size(0)) ) );
    shiftY = (int)( (new_cube_origin_meters(1) - origin_metric(1)) * ( volume_resolution(1) / (float) (volume_size(1)) ) );
    shiftZ = (int)( (new_cube_origin_meters(2) - origin_metric(2)) * ( volume_resolution(2) / (float) (volume_size(2)) ) );

    std::cout << "SHIFT: " << "shiftX: " << shiftX << "shiftY: " << shiftY << "shiftZ: " << shiftZ << std::endl;

    // update the cube's metric origin
    origin_metric = new_cube_origin_meters;
}

void cvpr_tum::CircularBuffer::clearTSDFSlice (TsdfVolume::TsdfVolumePtr tsdf_volume, int shiftX, int shiftY, int shiftZ)
{
    /*
    for (unsigned int x = minBounds_(0); x < maxBounds_(0); x++) {
        for (unsigned int y = minBounds_(1); y < maxBounds_(1); y++) {
            for (unsigned int z = minBounds_(2); z < maxBounds_(2); ++z) {
                //idx_curr = volume_resolution(0) * (z * volume_resolution(1) + y) + x;
                tsdf_volume->setv(x,y,z, 1.f);
            }
        }
    }
    */
    unsigned int idx_curr = 0;
    for (unsigned int z = 0; z < volume_resolution(2); z++) {
        for (unsigned int y = 0; y < volume_resolution(1); y++) {
            for (unsigned int x = 0; x < volume_resolution(0); x++) {
                if ((x >= minBounds_(0) && x < maxBounds_(0)) || (y >= minBounds_(1) && y < maxBounds_(1)) || (z >= minBounds_(2) && z < maxBounds_(2)))
                {
                    idx_curr = volume_resolution(0) * (z * volume_resolution(1) + y) + x;
                    float* pos_value = tsdf_volume->begin() + idx_curr;
                    shift_tsdf_pointer(&pos_value);
                    *pos_value = tsdf_volume->getPositiveTsdfTruncDist();
                }
            }
        }
    }
}

void cvpr_tum::CircularBuffer::computeMinMaxBounds(int newX, int newY, int newZ)
{
    //X
    if (newX >= 0)
    {
        minBounds_(0) = origin_GRID(0);
        maxBounds_(0) = newX;
    }
    else
    {
        minBounds_(0) = newX + volume_resolution(0) - 1;
        maxBounds_(0) = origin_GRID(0) + volume_resolution(0) - 1;
    }

    if (minBounds_(0) > maxBounds_(0))
        std::swap (minBounds_(0), maxBounds_(0));

    //Y
    if (newY >= 0)
    {
        minBounds_(1) = origin_GRID(1);
        maxBounds_(1) = newY;
    }
    else
    {
        minBounds_(1) = newY + volume_resolution(1) - 1;
        maxBounds_(1) = origin_GRID(1) + volume_resolution(1) - 1;
    }

    if(minBounds_(1) > maxBounds_(1))
        std::swap (minBounds_(1), maxBounds_(1));

    //Z
    if (newZ >= 0)
    {
        minBounds_(2) = origin_GRID(2);
        maxBounds_(2) = newZ;
    }
    else
    {
        minBounds_(2) = newZ + volume_resolution(2) - 1;
        maxBounds_(2) = origin_GRID(2) + volume_resolution(2) - 1;
    }

    if (minBounds_(2) > maxBounds_(2))
        std::swap(minBounds_(2), maxBounds_(2));

    minBounds_(0) -= origin_GRID(0);
    maxBounds_(0) -= origin_GRID(0);

    minBounds_(1) -= origin_GRID(1);
    maxBounds_(1) -= origin_GRID(1);

    minBounds_(2) -= origin_GRID(2);
    maxBounds_(2) -= origin_GRID(2);

    if (minBounds_(0) < 0) // We are shifting Left
    {
        minBounds_(0) += volume_resolution(0);
        maxBounds_(0) += (volume_resolution(0));
    }

    if (minBounds_(1) < 0) // We are shifting up
    {
        minBounds_(1) += volume_resolution(1);
        maxBounds_(1) += (volume_resolution(1));
    }

    if (minBounds_(2) < 0) // We are shifting back
    {
        minBounds_(2) += volume_resolution(2);
        maxBounds_(2) += volume_resolution(2);
    }
    std::cout << "minBounds_: " << minBounds_(0) << " " << minBounds_(1) << " " << minBounds_(2) << std::endl;
    std::cout << "maxBounds_: " << maxBounds_(0) << " " << maxBounds_(1) << " " << maxBounds_(2) << std::endl;
}
