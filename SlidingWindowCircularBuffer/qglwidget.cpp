#include <QtGui/QMouseEvent>
#include <QTimer>
#include "glwidget.h"

float GLWidget::map[MAP_SIZEX][MAP_SIZEY] = {
  {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3, 1, 0, 0, 0.5},
  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.5},
  {1, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0, 1, 0, 0, 0.5},
  {1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0.5},
  {1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.5},
  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.5},
  {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.5},
  {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.5},
  {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.5},
  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0.5}
};

GLWidget::GLWidget(QWidget *parent)
    : QGLWidget(parent),
      m_pos( 0.0f, 0.0f ),
      m_size( VIEW_SIZEX*WIDTH/MAP_SIZEX, VIEW_SIZEY*HEIGHT/MAP_SIZEY ),
      m_velocity( 1.0f, 1.0f ),
      m_tile_size( WIDTH/MAP_SIZEX, HEIGHT/MAP_SIZEY ),
      m_aspectRatio( 1.0 )
{
    setMouseTracking(true);
    QTimer* timer = new QTimer( this );
    connect( timer, SIGNAL( timeout() ), SLOT( step() ) );
    timer->start( 1000 / 1 );
    buffer_ = CircularBuffer::CircularBufferSharedPtr(new CircularBuffer());
    buffer_->resetBuffer();
}

void GLWidget::initializeGL() {
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glEnable(GL_POLYGON_SMOOTH);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Set the clear color to blue
    glClearColor( 0.0f, 0.0f, 1.0f, 1.0f );
}

void GLWidget::resizeGL(int w, int h) {
    // Set the viewport to window dimensions
    glViewport( 0, 0, w, h );

    // reset the coordinate system
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    glOrtho( 0.0, w, h, 0.0, 0.0, -1.0 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();
}

void GLWidget::paintGL() {
    // Clear the buffer with the current clearing color
    glClear( GL_COLOR_BUFFER_BIT );

    for (int y = 0; y < MAP_SIZEY; y++)
    {
        for (int x = 0; x < MAP_SIZEX; x++)
        {
            glColor3f( GLWidget::map[x][y], GLWidget::map[x][y], GLWidget::map[x][y] );
            //if (GLWidget::map[x][y]) glColor3f( 0.0f, 0.0f, 0.0f );
            glRectf( m_tile_size.x()*x, m_tile_size.y()*y, m_tile_size.x()*(x + 1), m_tile_size.y()*(y+1) );
            /*
            glBegin(GL_LINE_LOOP);
            glVertex2f(m_tile_size.x()*x, m_tile_size.y()*y);
            glVertex2f(m_tile_size.x()*(x + 1), m_tile_size.y()*y);
            glVertex2f(m_tile_size.x()*(x + 1), m_tile_size.y()*(y+1));
            glVertex2f(m_tile_size.x()*x, m_tile_size.y()*(y+1));
            glEnd();
            */
        }
    }
    // Set drawing colour to red
    glColor3f( 1.0f, 0.0f, 0.0f );

    // Draw a filled rectangle with the current color
    //glRectf( m_pos.x(), m_pos.y(), m_pos.x() + m_size.x(), m_pos.y() + m_size.y() );
    glBegin(GL_LINE_LOOP);
    glVertex2f(m_pos.x(), m_pos.y());
    glVertex2f( m_pos.x() + m_size.x(), m_pos.y());
    glVertex2f( m_pos.x() + m_size.x(), m_pos.y() + m_size.y());
    glVertex2f(m_pos.x(), m_pos.y() + m_size.y());
    glEnd();
}

void GLWidget::mousePressEvent(QMouseEvent *event) {

}
void GLWidget::mouseMoveEvent(QMouseEvent *event) {
    printf("%d, %d\n", event->x(), event->y());
}

void GLWidget::keyPressEvent(QKeyEvent* event) {
    switch(event->key()) {
    case Qt::Key_Escape:
        close();
        break;
    default:
        event->ignore();
        break;
    }
}

void GLWidget::step()
{
    updateGL();

    float windowWidth = WIDTH;
    float windowHeight = HEIGHT;

    if ( m_aspectRatio >= 1.0 )
        windowWidth = WIDTH * m_aspectRatio;
    else
        windowHeight = HEIGHT / m_aspectRatio;

    QVector2D shift = m_tile_size * m_velocity;
    // Reverse direction when we reach the left or right edge
    if ( m_pos.x() > windowWidth - m_size.x() - m_tile_size.x() || m_pos.x() + shift.x() < 0 )
        m_velocity.setX( -m_velocity.x() );

    // Reverse direction when we reach the top or bottom edge
    if ( m_pos.y() > windowHeight - m_size.y() - m_tile_size.y() || m_pos.y() + shift.y() < 0 )
        m_velocity.setY( -m_velocity.y() );

    buffer_->checkForShift(m_pos, true);

    //integrate
    buffer_->integrate( QVector2D(m_pos.x() / (WIDTH/MAP_SIZEX), m_pos.y() / (HEIGHT/MAP_SIZEY)), GLWidget::map );

    emitsig();

    // Update the position
    m_pos += m_tile_size * m_velocity;

/*
    // Make sure that the rectangle is always within the window,
    // even if it gets resized at an inappropriate time
    if ( m_pos.x() > ( windowWidth - m_size.x() + m_velocity.x() ) )
        m_pos.setX( windowWidth - m_size.x() - m_tile_size.x() );

    if ( m_pos.y() > ( windowHeight - m_size.y() + m_velocity.y() ) )
        m_pos.setY( windowHeight - m_size.y() - m_tile_size.y() );
*/
}

void GLWidget::emitsig()
{
    emit dataIntegratedSignal();
}
