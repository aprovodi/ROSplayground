#include "volumewidget.h"

VolumeWidget::VolumeWidget(const CircularBuffer::CircularBufferSharedPtr& buffer, QWidget *parent)
    : buffer_(buffer), QGLWidget(parent)
{
}

void VolumeWidget::initializeGL() {
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glEnable(GL_POLYGON_SMOOTH);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Set the clear color to blue
    glClearColor( 0.0f, 0.0f, 1.0f, 1.0f );
}

void VolumeWidget::resizeGL(int w, int h) {
    // Set the viewport to window dimensions
    glViewport( 0, 0, w, h );

    // reset the coordinate system
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    glOrtho( 0.0, w, h, 0.0, 0.0, -1.0 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
}

void VolumeWidget::paintGL() {
    // Clear the buffer with the current clearing color
    glClear( GL_COLOR_BUFFER_BIT );

    if (!buffer_) return;
    unsigned int idx_curr = 0;
    QVector2D m_tile_size( WIDTH/MAP_SIZEX, HEIGHT/MAP_SIZEY );
    for (int x = 0; x < VOLUME_RESX; x++)
    {
        for (int y = 0; y < VOLUME_RESY; y++)
        {
            idx_curr = VOLUME_RESX * y + x;
            float * value = buffer_->tsdf_data(idx_curr);
            buffer_->shift_tsdf_pointer(&value);
            glColor3f( *value, *value, *value );
            glRectf( m_tile_size.x()*x, m_tile_size.y()*y, m_tile_size.x()*(x + 1), m_tile_size.y()*(y+1) );
        }
    }
}

void VolumeWidget::update()
{
    updateGL();
}
