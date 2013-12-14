#include "circularbuffer.h"

bool CircularBuffer::checkForShift (const QVector2D &cam_pose, const bool perform_shift)
{
    bool result = false;

    // project the target point in the cube
    QVector2D targetPoint;
    targetPoint.setX(0.0);
    targetPoint.setY(0.0); // place the point at camera position + distance_camera_target on Z
    //targetPoint += volume_size / 2;
    targetPoint = cam_pose + targetPoint;

//    std::cout << "targetPoint: " << targetPoint << std::endl;

    // check distance from the cube's center
    QVector2D center_cube;
    center_cube = origin_metric + volume_size / 2.0f;

//    std::cout << "origin_metric: " << origin_metric << std::endl;

//    std::cout << "euclideanDistance: " << (targetPoint - center_cube).norm() << std::endl;

//    if ((targetPoint - center_cube).length() > distance_threshold_)
    if (targetPoint.x() < origin_metric.x() || targetPoint.y() < origin_metric.y() ||
            targetPoint.x() + VIEW_SIZEX*WIDTH/MAP_SIZEX > origin_metric.x()+volume_size.x() || targetPoint.y() + VIEW_SIZEY*HEIGHT/MAP_SIZEY > origin_metric.y() + volume_size.y())
        result = true;

    if (!perform_shift)
        return (result);

    // perform shifting operations
    if (result)
        performShift (targetPoint);

    return (result);
}

void CircularBuffer::performShift (const QVector2D &target_point)
{
    // compute new origin and offsets
    int offset_X, offset_Y;
    computeAndSetNewCubeMetricOrigin (target_point, offset_X, offset_Y);

    int newX = origin_GRID.x() + offset_X;
    int newY = origin_GRID.y() + offset_Y;

    computeMinMaxBounds(newX, newY);


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
    clearTSDFSlice (offset_X, offset_Y);

    // shift buffer addresses
    shiftOrigin (QVector2D(offset_X, offset_Y));

    /*
    // push existing data in the TSDF buffer
    if (previously_existing_slice->points.size () != 0 ) {
      volume->pushSlice(previously_existing_slice, getBuffer () );
    }
    */
}

void CircularBuffer::computeAndSetNewCubeMetricOrigin (const QVector2D &target_point, int &shiftX, int &shiftY)
{
    // compute new origin for the cube, based on the target point
    QVector2D new_cube_origin_meters;
    new_cube_origin_meters = target_point;// - volume_size / 2.0;
    //std::cout << "The old cube's metric origin was: " << origin_metric << std::endl;
    //std::cout << "The new cube's metric origin is now: " << new_cube_origin_meters << std::endl;

    // deduce each shift in indices
    shiftX = (int)( (new_cube_origin_meters.x() - origin_metric.x()) * ( volume_resolution.x() / (float) (volume_size.x()) ) );
    shiftY = (int)( (new_cube_origin_meters.y() - origin_metric.y()) * ( volume_resolution.y() / (float) (volume_size.y()) ) );

    // update the cube's metric origin
    origin_metric = new_cube_origin_meters;
}

void CircularBuffer::clearTSDFSlice (int shiftX, int shiftY)
{
    unsigned int idx_curr = 0;
    for (unsigned int x = 0; x < volume_resolution.x(); x++) {
        for (unsigned int y = 0; y < volume_resolution.y(); y++) {
            if ((x >= minBoundsx_ && x < maxBoundsx_) || (y >= minBoundsy_ && y < maxBoundsy_))
            {
                idx_curr = volume_resolution.x()*y + x;
                float* pos_value = tsdf_ + idx_curr;
                // shift the pointer to relative indices
                shift_tsdf_pointer(&pos_value);
                *pos_value = 0.0;
            }
        }
    }
}

void CircularBuffer::computeMinMaxBounds(int newX, int newY)
{
    //X
    if (newX >= 0)
    {
        minBoundsx_ = origin_GRID.x();
        maxBoundsx_ = newX;
    }
    else
    {
        minBoundsx_ = newX + volume_resolution.x() - 1;
        maxBoundsx_ = (float)origin_GRID.x() + volume_resolution.x() - 1;
    }

    if (minBoundsx_ > maxBoundsx_)
        std::swap (minBoundsx_, maxBoundsx_);

    //Y
    if (newY >= 0)
    {
        minBoundsy_ = origin_GRID.y();
        maxBoundsy_ = newY;
    }
    else
    {
        minBoundsy_ = newY + volume_resolution.y() - 1;
        maxBoundsy_ = origin_GRID.y() + volume_resolution.y() - 1;
    }

    if(minBoundsy_ > maxBoundsy_)
        std::swap (minBoundsy_, maxBoundsy_);

    minBoundsx_ -= origin_GRID.x();
    maxBoundsx_ -= origin_GRID.x();

    minBoundsy_ -= origin_GRID.y();
    maxBoundsy_ -= origin_GRID.y();

    if (minBoundsx_ < 0) // We are shifting Left
    {
        minBoundsx_ += volume_resolution.x();
        maxBoundsx_ += volume_resolution.x();
    }

    if (minBoundsy_ < 0) // We are shifting up
    {
        minBoundsy_ += volume_resolution.y();
        maxBoundsy_ += volume_resolution.y();
    }
}
