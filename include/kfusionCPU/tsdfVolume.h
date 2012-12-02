#ifndef KFUSIONCPU_TSDF_VOLUME_H_
#define KFUSIONCPU_TSDF_VOLUME_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <utility>
#include "kfusionCPU/internal.hpp"

namespace cvpr_tum
{
class TsdfVolume
{
    enum Weights
    {
        MAX_WEIGHT = 1 << 7
    };
public:
    typedef boost::shared_ptr<TsdfVolume> TsdfVolumePtr;
    typedef float* iterator;

    /** \brief Constructor
     * \param[in] resolution volume resolution (number of voxels)
     */
    TsdfVolume(const Eigen::Vector3i& resolution);

    /** \brief Destructor */
    ~TsdfVolume();

    /** \brief release volume data manually */
    void release();

    /** \brief Sets tsdf volume data to initial state */
    void reset();

    /** \brief Iterators for the volume */
    iterator begin();
    iterator end();

    /** \brief Sets Tsdf volume size for each dimention
     * \param[in] size size of tsdf volume in meters
     * */
    void setSize(const Eigen::Vector3f& size);

    /** \brief Sets Tsdf truncation distance. Must be greater than 2 * volume_voxel_size
     * \param[in] distance TSDF truncation distance
     * */
    void setPositiveTsdfTruncDist(float distance);
    void setNegativeTsdfTruncDist(float distance);

    /** \brief Returns volume size in meters */
    const Eigen::Vector3f& getSize() const;

    /** \brief Returns volume resolution */
    const Eigen::Vector3i& getResolution() const;

    float getPositiveTsdfTruncDist() const;
    float getNegativeTsdfTruncDist() const;

    /** \brief Returns volume voxel size in meters */
    const Eigen::Vector3f& getVoxelSize() const;

    /** \brief Returns tsdf value from a specific location */
    float getInterpolatedTSDFValue(const Eigen::Vector3f& glocation);

    /** \brief Returns gradient value from a specific location
     * \param[in] glocation world coo of the point
     * */
    Eigen::Matrix<float, 1, 3> getTSDFGradient(const Eigen::Vector3f& glocation);

    /** \brief Integrates new data to the current volume
     * \param[in] raw_depth_map Depth data to integrate
     * \param[in] intr Camera intrinsics
     * \param[in] camtoworld Camera transformations
     * */
    void integrate(const cv::Mat& raw_depth_map, const Intr& intr, const Eigen::Matrix3f& Rcam_inv,
                   const Eigen::Vector3f& tcam);

    /** \brief Returns tsdf value from a specific location
     * \param[in] pi ith voxel coordinate in a grid
     * \param[in] pj jth voxel coordinate in a grid
     * \param[in] pk kth voxel coordinate in a grid
     * */
    float v(int pi, int pj, int pk) const;

    /** \brief Tells if the gradient from a specific location is valid*/
    bool validGradient(const Eigen::Vector3f& glocation);

private:

    /** \brief tsdf volume data */
    float* tsdf_;
    float* weights_;

    /** \brief tsdf volume size in meters */
    Eigen::Vector3f world_size_;

    /** \brief tsdf volume resolution */
    Eigen::Vector3i resolution_;

    /** \brief tsdf truncation distance */
    float pos_tranc_dist_;

    /** \brief tsdf truncation distance */
    float neg_tranc_dist_;

    /** \brief tsdf cell size */
    Eigen::Vector3f cell_size_;

    /** \brief tsdf number of cell */
    unsigned int num_cells_;

    unsigned int num_columns_;
    unsigned int num_rows_;
    unsigned int num_slices_;

    /** \brief Calculates global coordinates of voxel
     * \param[in] x voxel coordinate in a grid
     * \param[in] y voxel coordinate in a grid
     * \param[in] z voxel coordinate in a grid
     */
    Eigen::Vector3f getVoxelGCoo(int x, int y, int z) const;

};
}

#endif /* KFUSIONCPU_TSDF_VOLUME_H_ */
