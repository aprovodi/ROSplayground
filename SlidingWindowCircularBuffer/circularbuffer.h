#ifndef CIRCULARBUFFER_H
#define CIRCULARBUFFER_H

#include <QVector2D>
#include <QSharedPointer>

#define VOLUME_RESX 9
#define VOLUME_RESY 9

#define MAP_SIZEX 10
#define MAP_SIZEY 14

#define VIEW_SIZEX 5
#define VIEW_SIZEY 5

#define WIDTH 600
#define HEIGHT 840

class CircularBuffer
{
public:
    typedef QSharedPointer<CircularBuffer> CircularBufferSharedPtr;

    CircularBuffer (const QVector2D vol_res = QVector2D(VOLUME_RESX, VOLUME_RESY),
                    const QVector2D vol_size = QVector2D(VOLUME_RESX*WIDTH/MAP_SIZEX, VOLUME_RESY*HEIGHT/MAP_SIZEY))
        : volume_size( vol_size ), volume_resolution( vol_res ), distance_threshold_(180)
    {
        num_cells_ = vol_res.x()*vol_res.y();
        tsdf_memory_start = 0; tsdf_memory_end = 0; tsdf_rolling_buff_origin = 0;
        tsdf_ = new float[num_cells_];
        std::fill(tsdf_, tsdf_ + num_cells_, 0);
    }
    ~CircularBuffer()
    {
        delete[] tsdf_;
        tsdf_ = NULL;
    }

    bool checkForShift (const QVector2D &cam_pose, const bool perform_shift = true);

    void performShift (const QVector2D &target_point);

    void computeAndSetNewCubeMetricOrigin (const QVector2D &target_point, int &shiftX, int &shiftY);

    /** \brief updates cyclical buffer origins given offsets on X, Y and Z
    * \param[in] tsdf_volume pointer to the TSDF volume managed by this cyclical buffer
    * \param[in] offset in indices
    */
    void shiftOrigin (const QVector2D offset)
    {
        // shift rolling origin (making sure they keep in [0 - NbVoxels[ )
        origin_GRID.setX(origin_GRID.x() + offset.x());
        if(origin_GRID.x() >= volume_resolution.x())
            origin_GRID.setX(origin_GRID.x() - volume_resolution.x());
        else if(origin_GRID.x() < 0)
            origin_GRID.setX(origin_GRID.x() + volume_resolution.x());

        origin_GRID.setY(origin_GRID.y() + offset.y());
        if(origin_GRID.y() >= volume_resolution.y())
            origin_GRID.setY(origin_GRID.y() - volume_resolution.y());
        else if(origin_GRID.y() < 0)
            origin_GRID.setY(origin_GRID.y() + volume_resolution.y());

        // update memory pointers
        tsdf_memory_start = tsdf_;
        tsdf_memory_end = tsdf_ + num_cells_;
        tsdf_rolling_buff_origin = tsdf_ + (int)(volume_resolution.x() * origin_GRID.y() + origin_GRID.x());

        // update global origin
        origin_GRID_global += offset;
    }

    void clearTSDFSlice( int shiftX, int shiftY );

    void integrate (const QVector2D& pos, float map[MAP_SIZEX][MAP_SIZEY])
    {
        unsigned int idx_curr = 0;
        for (unsigned int x = 0; x < volume_resolution.x(); x++) {
            for (unsigned int y = 0; y < volume_resolution.y(); y++) {
                    QVector2D v_g(getVoxelGCoo(x, y));

                    idx_curr = volume_resolution.x() * v_g.y() + v_g.x();

                    float* pos_value = tsdf_ + idx_curr;
                    // shift the pointer to relative indices
                    shift_tsdf_pointer(&pos_value);

                    //QVector2D coo((x + (int)origin_GRID.x() - (int)pos.x()) % (int)volume_resolution.x(), (y + (int)origin_GRID.y() - (int)pos.y()) % (int)volume_resolution.y());
                    QVector2D coo(x + origin_GRID_global.x() - pos.x(), y + origin_GRID_global.y() - pos.y());
                    if (coo.x() < 0 || coo.y() < 0 || coo.x() >= VIEW_SIZEX || coo.y() >= VIEW_SIZEY)
                        continue;

                    *pos_value = map[(int)(x + origin_GRID_global.x())][(int)(y + origin_GRID_global.y())];
            }
        }
    }

    QVector2D getVoxelGCoo(int x, int y) const {
        int ray_x = x;//(x + volume_resolution.x() / 2 - (VOLUME_RESX - VIEW_SIZEX)/2);
        int ray_y = y;//(y + volume_resolution.y() / 2 - (VOLUME_RESY - VIEW_SIZEY)/2);
        return QVector2D(ray_x, ray_y);
    }

    void shift_tsdf_pointer(float** value)
    {
        ///Shift the pointer by (@origin - @start)
        int shift = tsdf_rolling_buff_origin - tsdf_memory_start;
        *value += (tsdf_rolling_buff_origin - tsdf_memory_start);

        ///If we land outside of the memory, make sure to "modulo" the new value
        if(*value > tsdf_memory_end)
        {
            *value -= (tsdf_memory_end - tsdf_memory_start + 1);
        }
    }

    void initBuffer ()
    {
        tsdf_memory_start = tsdf_;
        tsdf_memory_end = tsdf_ + num_cells_;
        tsdf_rolling_buff_origin = tsdf_memory_start;
    }

    /** \brief Reset buffer structure
    * \param[in] tsdf_volume pointer to the TSDF volume managed by this cyclical buffer
    */
    void resetBuffer ()
    {
        origin_GRID = QVector2D();
        origin_GRID_global = QVector2D();
        origin_metric = QVector2D();
        initBuffer ();
    }

    /* Sets the distance threshold between cube's center and target point that triggers a shift. */
    void setDistanceThreshold (const float threshold) { distance_threshold_ = threshold; }

    /* Set the physical size represented by the default TSDF volume. */
    void setVolumeSize(const QVector2D& size) { volume_size = size; }

    inline float* tsdf_data() { return tsdf_; }
    inline float* tsdf_data(int pos) { return tsdf_ + pos; }

    inline const float* tsdf_start() { return tsdf_memory_start; }

    inline const float* tsdf_end() { return tsdf_memory_end; }

    inline const QVector2D& get_current_origin () { return origin_metric; }

    void computeMinMaxBounds(int newX, int newY);

private:

    float* tsdf_;
    float distance_threshold_;
    float pos_tranc_dist_;

    int num_cells_;

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
    QVector2D origin_GRID;
    /** \brief Cube origin in world coordinates.*/
    QVector2D origin_GRID_global;
    /** \brief Current metric origin of the cube, in world coordinates.*/
    QVector2D origin_metric;

    QVector2D volume_size;
    QVector2D volume_resolution; //512

    /** new shifting boundaries **/
    float minBoundsx_, minBoundsy_;
    float maxBoundsx_, maxBoundsy_;

};

#endif
