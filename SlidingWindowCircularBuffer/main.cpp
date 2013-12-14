#include <QApplication>
#include "GLWidget.h"
#include "volumewidget.h"
#include "mainwindow.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    CircularBuffer::CircularBufferSharedPtr buffer = CircularBuffer::CircularBufferSharedPtr(new CircularBuffer());
/*
    GLWidget window = GLWidget();
    window.initBuffer( buffer );
    window.resize(800,600);
    window.show();

    VolumeWidget volume(buffer->tsdf_data());
    volume.resize(640,420);
    volume.show();
*/
    MainWindow w;
    w.show();
    return a.exec();
}
