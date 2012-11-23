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
    typedef TsdfCell* iterator;

    /** \brief Constructor
     * \param[in] resolution volume resolution
     */
    TsdfVolume(const Eigen::Vector3i& resolution);

    /** \brief Destructor */
    ~TsdfVolume();

    /** \brief Resets tsdf volume data to uninitialized state */
    void reset();

    /** \brief Iterators for the volume */
    iterator begin();
    iterator end();

    /** \brief Sets Tsdf volume size for each dimention
     * \param[in] size size of tsdf volume in meters
     */
    void setSize(const Eigen::Vector3f& size);

    /** \brief Sets Tsdf truncation distance. Must be greater than 2 * volume_voxel_size
     * \param[in] distance TSDF truncation distance
     */
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
    float getTSDFValue(const Eigen::Vector3f& glocation);

    /** \brief Returns gradient value from a specific location */
    float getTSDFGradient(const Eigen::Vector3f& glocation, int stepSize, int dim);

    /** \brief Tells if the gradient from a specific location is valid*/
    bool validGradient(const Eigen::Vector3f& glocation);

    /** \brief Integrates new data to the current volume
     * \param[in] raw_depth_map Depth data to integrate
     * \param[in] intr Camera intrinsics
     * \param[in] camtoworld Camera transformations
     */
    void integrate(const cv::Mat& raw_depth_map, const Intr& intr, const Eigen::Matrix4f& camtoworld);

private:

    /** \brief tsdf volume data */
    TsdfCell* tsdf_;

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

    /** \brief tsdf number of cell */
    unsigned int num_cells_;

    /** \brief Calculates global coordinates of voxel
     * \param[in] x voxel coordinate
     * \param[in] y voxel coordinate
     * \param[in] z voxel coordinate
     */
    Eigen::Vector3f getVoxelGCoo(int x, int y, int z) const;

    /** \brief Returns cell from a specific location */
    TsdfCell& operator[](const Eigen::Vector3i& position) const;

    /** \brief Returns tsdf value from a specific location */
    float v(int pi, int pj, int pk) const;

    /** \brief Sets TsdfCell to a specific location */
    void set(int px, int py, int pz, const TsdfCell& d);
};
}

#endif /* KFUSIONCPU_TSDF_VOLUME_H_ */
