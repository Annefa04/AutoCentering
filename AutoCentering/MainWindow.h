
#pragma once

#include <QtWidgets/QMainWindow>

#include <QImage>
#include <QVariantAnimation>
#include <QFutureWatcher>
#include <QString>
#include <QStringList>
#include <QDir>
#include <QFileInfo>
#include <QElapsedTimer>

#include <opencv2/opencv.hpp>

#include "ui_MainWindow.h"
#include "YOLODetector.h"
#include "OpenCVOnlyGeometry.h"

#include <QtConcurrent/QtConcurrent>

class QComboBox;
class QLabel;
class QPushButton;

class AutoCentering : public QMainWindow
{
    Q_OBJECT

public:
    AutoCentering(QWidget* parent = nullptr);
    ~AutoCentering();

private slots:

    // Single-image processing
    void loadImage();
    void processImage();
    void processOpenCVOnly();
    void processYOLOGeometricRefinement();
    void autoCenterImage();

    // Batch processing
    void browseBatchFolder();
    void startBatchAutoMode();


private:

    // ========================================================
    // DETECTION METHOD
    // ========================================================

    enum class DetectionMethod
    {
        None,
        YOLO,
        GeometricRefinement,
        YOLOGeometricRefinement
    };

    DetectionMethod activeMethod = DetectionMethod::None;

    // ========================================================
    // REUSABLE CENTERING RESULT
    // ========================================================

    struct CenteringResult
    {
        bool success = false;

        cv::Mat correctedImage;
        cv::Mat previewImage;

        double rotationCorrection = 0.0;

        double translationX = 0.0;
        double translationY = 0.0;

        double finalOffsetX = 0.0;
        double finalOffsetY = 0.0;

        double confidence = -1.0;

        QString errorMessage;
    };

    // ========================================================
    // BATCH AUTO MODE
    // ========================================================

    void setupBatchAutoMode();

    void processBatchImages(
        const QStringList& imagePaths,
        const QString& outputDirectory,
        DetectionMethod selectedMethod
    );

    CenteringResult processImageForCentering(
        const cv::Mat& inputImage,
        DetectionMethod selectedMethod,
        const QString& diagnosticDirectory
    );

    QString getMethodName(
        DetectionMethod method
    ) const;

    QString getMethodSuffix(
        DetectionMethod method
    ) const;

    // ========================================================
    // UI
    // ========================================================

    Ui::AutoCenteringClass ui;

    QComboBox* cmbBatchMethod = nullptr;
    QLabel* lblBatchFolder = nullptr;
    QPushButton* btnBrowseBatchFolder = nullptr;
    QPushButton* btnStartBatch = nullptr;

    QString batchInputDirectory;

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
    // ANIMATION
    // ========================================================

    QVariantAnimation* shiftAnimation = nullptr;
};
