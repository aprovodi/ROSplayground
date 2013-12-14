#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    a = new GLWidget(ui->centralwidget);
    a->setGeometry(QRect(10, 10, 600, 840));
    b = new VolumeWidget(a->getBuf(), ui->centralwidget);
    b->setGeometry(QRect(820, 10, 540, 540));
    //connect(this,SIGNAL(TosignalA()), this->a, SLOT(emitsig()));
    connect(this->a, SIGNAL(dataIntegratedSignal()), this, SLOT(emitB()));
    connect(this, SIGNAL(TosignalB()), this->b, SLOT(update()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::emitB()
{
    emit TosignalB();
}
/*
void MainWindow::on_pushButton_clicked()
{
    emit TosignalA();
}
*/
