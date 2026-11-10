#pragma once

#include <QtWidgets/QMainWindow>
#include <QImage>
#include <QVariantAnimation>
#include <QFutureWatcher>
#include <QString>

#include <opencv2/opencv.hpp>

#include "ui_MainWindow.h"
#include "YOLODetector.h"
#include "OpenCVOnlyGeometry.h"

#include <QtConcurrent/QtConcurrent>

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
    void processOpenCVOnly();

    // YOLO + Geometric Refinement detection only
    void processYOLOGeometricRefinement();

private:

    // ========================================================
    // ACTIVE DETECTION METHOD
    // ========================================================

    enum class DetectionMethod
    {
        None,
        YOLO,
        GeometricRefinement
    };

    DetectionMethod activeMethod =
        DetectionMethod::None;

    // ========================================================
    // UI
    // ========================================================

    Ui::AutoCenteringClass ui;

    // ========================================================
    // IMAGE DATA
    // ========================================================

    QString currentImagePath;
    QImage currentImage;

    // ========================================================
    // DETECTION
    // ========================================================

    QFutureWatcher<OpenCVOnlySegmentationResult>* openCVWatcher =
        nullptr;

    YOLODetector* detector = nullptr;

    // ========================================================
    // AUTO CENTER ANIMATION
    // ========================================================

    QVariantAnimation* shiftAnimation = nullptr;
};