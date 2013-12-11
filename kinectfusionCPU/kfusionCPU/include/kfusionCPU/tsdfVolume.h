#ifndef INCLUDE_KFUSIONCPU_TSDFVOLUME_H_
#define INCLUDE_KFUSIONCPU_TSDFVOLUME_H_

#include <Eigen/Geometry>
#include <boost/shared_ptr.hpp>

#include "kfusionCPU/internal.hpp"

namespace cvpr_tum {

class CircularBuffer;

class TsdfVolume {
    enum Weights { MAX_WEIGHT = 1 << 7 };

    public:
        // should be multiple of 32
        enum VolumeResolution { VOLUME_X = 64, VOLUME_Y = 64, VOLUME_Z = 64 };

        typedef boost::shared_ptr<TsdfVolume> TsdfVolumePtr;
        typedef boost::shared_ptr<const TsdfVolume> TsdfVolumeConstPtr;

        typedef float* iterator;
        typedef const float* const_iterator;

        /** \brief Constructor
         * \param[in] resolution volume resolution (number of voxels)
         */
        explicit TsdfVolume(const Eigen::Vector3i& resolution);

        /** \brief Destructor */
        ~TsdfVolume();

        /** \brief release volume data manually */
        void release();

        /** \brief Sets tsdf volume data to initial state */
        void reset();

        /** \brief Initializes with user allocated buffer.
         * \param ptr_arg: pointer to buffer
         * \param size_arg: data size
         * */
        void copyFrom(float *ptr_arg, unsigned int data_size_arg);

        inline float* data() { return tsdf_; }
        inline float* weights() { return weights_; }

        inline const float* data() const { return tsdf_; }
        inline const float* weights() const { return weights_; }

        /** \brief Iterators for the volume */
        inline iterator begin() { return tsdf_; }
        inline const_iterator begin() const { return tsdf_; }
        inline iterator begin_weights() { return weights_; }
        inline const_iterator begin_weights() const { return weights_; }

        inline iterator end() { return tsdf_ + num_cells_; }
        inline const_iterator end() const { return tsdf_ + num_cells_; }
        inline iterator end_weights() { return weights_ + num_cells_; }
        inline const_iterator end_weights() const { return weights_ + num_cells_; }

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
        const Eigen::Vector3f& getSize() const { return world_size_; }

        /** \brief Returns volume resolution */
        const Eigen::Vector3i& getResolution() const { return resolution_; }

        float getPositiveTsdfTruncDist() const { return pos_tranc_dist_; }
        float getNegativeTsdfTruncDist() const { return neg_tranc_dist_; }

        /** \brief Returns volume voxel size in meters */
        const Eigen::Vector3f& getVoxelSize() const { return cell_size_; }

        /** \brief Returns volume number of voxels */
        unsigned int getNumVoxels() const { return num_cells_; }

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
        void integrate(const cv::Mat& raw_depth_map,
                       const Intr& intr,
                       const Eigen::Matrix3f& Rcam_inv,
                       const Eigen::Vector3f& tcam, CircularBuffer* buffer);

        /** \brief Returns tsdf value from a specific location
         * \param[in] pi ith voxel coordinate in a grid
         * \param[in] pj jth voxel coordinate in a grid
         * \param[in] pk kth voxel coordinate in a grid
         * */
        float v(int pi, int pj, int pk) const;

        void setv(int pos_x, int pos_y, int pos_z, float val);

        /** \brief Tells if the gradient from a specific location is valid*/
        bool validGradient(const Eigen::Vector3f& glocation);

    private:
        /** \brief tsdf volume data */
        float* tsdf_;   // make it probably like a vector
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
        Eigen::Vector3f inverse_cell_size_;

        /** \brief tsdf number of cell */
        std::size_t num_cells_;

        std::size_t num_columns_;
        std::size_t num_rows_;
        std::size_t num_slices_;

        /** \brief Calculates global coordinates of voxel
         * \param[in] x voxel coordinate in a grid
         * \param[in] y voxel coordinate in a grid
         * \param[in] z voxel coordinate in a grid
         */
        Eigen::Vector3f getVoxelGCoo(int x, int y, int z) const;
    };
}  // namespace cvpr_tum

#endif  // INCLUDE_KFUSIONCPU_TSDFVOLUME_H_
