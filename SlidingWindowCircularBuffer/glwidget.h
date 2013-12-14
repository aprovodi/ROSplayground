#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>
#include <QVector2D>

#include "circularbuffer.h"

class GLWidget : public QGLWidget
{

    Q_OBJECT // must include this if you use Qt signals/slots

public:
    GLWidget(QWidget *parent = NULL);

    void initBuffer(const CircularBuffer::CircularBufferSharedPtr& buffer) { buffer_ = buffer; }
    CircularBuffer::CircularBufferSharedPtr getBuf() {return buffer_;}
protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void keyPressEvent(QKeyEvent *event);
private slots:
    void step();

signals:
    void dataIntegratedSignal();

public slots:
    void emitsig();

private:
    QVector2D m_pos;
    QVector2D m_size;
    QVector2D m_velocity;
    double m_aspectRatio;
    static float map[MAP_SIZEX][MAP_SIZEY];
    QVector2D m_tile_size;

    CircularBuffer::CircularBufferSharedPtr buffer_;
};

#endif // GLWIDGET_H
