#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "glwidget.h"
#include "volumewidget.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    GLWidget *a;
    VolumeWidget *b;

signals:
//    void TosignalA();
    void TosignalB();

public slots:
    void emitB();

//private slots:
//    void on_pushButton_clicked();
};

#endif // MAINWINDOW_H
