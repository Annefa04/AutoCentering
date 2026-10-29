#pragma once

#include <QtWidgets/QMainWindow>
#include <QImage>

#include "ui_MainWindow.h"
#include "YOLODetector.h"

#include <QVariantAnimation>

class AutoCentering : public QMainWindow
{
    Q_OBJECT

public:
    AutoCentering(QWidget* parent = nullptr);
    ~AutoCentering();

private slots:
    void loadImage();
    void processImage();
    void autoCenterImage();

private:
    Ui::AutoCenteringClass ui;

    QString currentImagePath;
    QImage currentImage;

    YOLODetector* detector = nullptr;
    QVariantAnimation* shiftAnimation = nullptr;
};

