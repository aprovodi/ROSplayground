#ifndef VOLUMEWIDGET_H
#define VOLUMEWIDGET_H

#include <QGLWidget>
#include <QVector2D>

#include "circularbuffer.h"

class VolumeWidget : public QGLWidget
{

public:
    VolumeWidget(const CircularBuffer::CircularBufferSharedPtr& buffer, QWidget *parent = NULL);
protected:
    void initializeGL();
    void resizeGL(int w, int h);
    void paintGL();

public slots:
    void update();

private:
    CircularBuffer::CircularBufferSharedPtr buffer_;
};

#endif // VOLUMEWIDGET_H
