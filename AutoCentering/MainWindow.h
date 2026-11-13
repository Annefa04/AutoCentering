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
#include <QProgressDialog>

#include <atomic>
#include <functional>

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

    // ========================================================
    // SINGLE-IMAGE PROCESSING
    // ========================================================

    void loadImage();

    void processImage();

    void processOpenCVOnly();

    void processYOLOGeometricRefinement();

    void autoCenterImage();


    // ========================================================
    // BATCH PROCESSING
    // ========================================================

    void browseBatchFolder();

    void startBatchAutoMode();


    // ========================================================
    // SAVE RESULT (single-image Auto Center)
    // ========================================================

    void saveResult();


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

        // Used to replay the original -> corrected animation.
        cv::Mat finalTransform;

        cv::RotatedRect initialBox;

        cv::Point2f initialCenter;

        QString errorMessage;
    };


    // ========================================================
    // BATCH SUMMARY (returned by the worker thread)
    // ========================================================

    struct BatchSummary
    {
        int total = 0;

        int processed = 0;

        int success = 0;

        int failed = 0;

        bool cancelled = false;

        QString methodName;

        QString outputDirectory;

        QString errorMessage;
    };


    // ========================================================
    // BATCH AUTO MODE
    // ========================================================

    void setupBatchAutoMode();

    void setProcessingState(bool processing);

    // Returns immediately. The result arrives through batchWatcher.
    void processBatchImages(
        const QStringList& imagePaths,
        const QString& outputDirectory,
        DetectionMethod selectedMethod
    );

    // GUI thread only. Non-blocking: starts a timer-driven animation
    // and clears batchAnimating when it finishes.
    void playBatchAnimation(
        const cv::Mat& originalSmall,
        const cv::Mat& transformSmall,
        const cv::RotatedRect& boxSmall,
        const cv::Point2f& centerSmall,
        int durationMs
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

    // Found by its text ("Save Result") in the constructor.
    QPushButton* btnSaveResult = nullptr;


    // ========================================================
    // BATCH STATE
    // ========================================================

    QString batchInputDirectory;

    // Background batch watcher.
    QFutureWatcher<BatchSummary>* batchWatcher = nullptr;

    // Cancellation flag shared with the worker thread.
    std::atomic<bool> batchCancelRequested{ false };

    // True while the GUI is playing the per-image animation.
    std::atomic<bool> batchAnimating{ false };


    // ========================================================
    // IMAGE DATA
    // ========================================================

    QString currentImagePath;

    QImage currentImage;

    // Result of the last single-image Auto Center. Held in memory
    // only; nothing is written to disk until "Save Result" is clicked.
    cv::Mat pendingCorrectedImage;

    cv::Mat pendingPreviewImage;

    QString pendingBaseName;


    // ========================================================
    // MANUAL AUTO CENTER (runs on a worker thread)
    // ========================================================

    QFutureWatcher<CenteringResult>* autoCenterWatcher = nullptr;

    // Per-run data used by the watcher's finished handler.
    cv::Mat autoCenterOriginal;

    QString autoCenterBaseName;

    QString autoCenterMethodName;


    // ========================================================
    // DETECTION
    // ========================================================

    QFutureWatcher<OpenCVOnlySegmentationResult>* openCVWatcher =
        nullptr;

    QFutureWatcher<YOLOGuidedRefinementResult>* hybridWatcher =
        nullptr;

    bool isProcessing = false;

    // YOLO confidence passed back from worker threads.
    std::atomic<float> lastYoloConfidence{ -1.0f };

    YOLODetector* detector = nullptr;


    // ========================================================
    // ANIMATION
    // ========================================================

    QVariantAnimation* shiftAnimation = nullptr;
};