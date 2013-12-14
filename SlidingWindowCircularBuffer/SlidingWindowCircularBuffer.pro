#-------------------------------------------------
#
# Project created by QtCreator 2013-11-23T21:05:45
#
#-------------------------------------------------

QT       += core gui opengl

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = SlidingWindowCircularBuffer
TEMPLATE = app


SOURCES += main.cpp \
    qglwidget.cpp \
    circularbuffer.cpp \
    volumewidget.cpp \
    mainwindow.cpp

HEADERS  += \
    glwidget.h \
    circularbuffer.h \
    volumewidget.h \
    mainwindow.h

INCLUDEPATH += C:/boost_1_54_0
#LIBS += -LC:/boost_1_54_0/stage/lib -lBoost_library

#LIBS += -lglut -lGL -lGLU -lGLEW

FORMS += \
    mainwindow.ui
