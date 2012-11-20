#ifndef KFUSIONCPU_TSDF_VOLUME_H_
#define KFUSIONCPU_TSDF_VOLUME_H_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>
#include <boost/multi_array.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <utility>

namespace cvpr_tum
{
class Intr;

class TsdfVolume
{
    enum Weights
    {
        MAX_WEIGHT = 1 << 7
    };
public:
    typedef boost::shared_ptr<TsdfVolume> Ptr;
    typedef std::pair<float, float> TsdfCell;
    typedef boost::multi_array<TsdfCell, 3> TsdfVolumeData;

    /** \brief Constructor
     * \param[in] resolution volume resolution
     */
    TsdfVolume(const Eigen::Vector3i& resolution);

    /** \brief Sets Tsdf volume size for each dimention
     * \param[in] size size of tsdf volume in meters
     */
    void setSize(const Eigen::Vector3f& size);

    /** \brief Sets Tsdf truncation distance. Must be greater than 2 * volume_voxel_size
     * \param[in] distance TSDF truncation distance
     */
    void setPositiveTsdfTruncDist(float distance);
    void setNegativeTsdfTruncDist(float distance);

    /** \brief Returns tsdf volume container */
    const TsdfVolumeData& data() const;

    /** \brief Returns volume size in meters */
    const Eigen::Vector3f& getSize() const;

    float getPositiveTsdfTruncDist() const;
    float getNegativeTsdfTruncDist() const;

    /** \brief Returns volume resolution */
    const Eigen::Vector3i& getResolution() const;

    /** \brief Returns volume voxel size in meters */
    const Eigen::Vector3f getVoxelSize() const;

    /** \brief Integrates TSDF Volume
     * \param[in] raw_depth_map
     * \param[in] intr Camera Intrinsics
     * \param[in] Rcam_inv Rotational part
     * \param[in] tcam Translational part
     */
    void integrate(const cv::Mat& raw_depth_map, const Intr& intr,
                   const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>& Rcam_inv, const Eigen::Vector3f& tcam);

private:

    /** \brief tsdf volume data */
    TsdfVolumeData tsdf_;

    /** \brief tsdf volume size in meters */
    Eigen::Vector3f size_;

    /** \brief tsdf volume resolution */
    Eigen::Vector3i resolution_;

    /** \brief tsdf truncation distance */
    float pos_tranc_dist_;

    /** \brief tsdf truncation distance */
    float neg_tranc_dist_;

    /** \brief tsdf cell size */
    Eigen::Vector3f cell_size_;

    /** \brief Calculates global coordinates of voxel
     * \param[in] x voxel coordinate
     * \param[in] y voxel coordinate
     * \param[in] z voxel coordinate
     */
    Eigen::Vector3f getVoxelGCoo(int x, int y, int z) const;

};
}

#endif /* KFUSIONCPU_TSDF_VOLUME_H_ */
