#include "MainWindow.h"
#include "OpenCVOnlyGeometry.h"

#include <QFile>
#include <QApplication>
#include <QCoreApplication>
#include <QDebug>
#include <QFileDialog>
#include <QPixmap>
#include <QMessageBox>
#include <QPainter>
#include <QPen>
#include <QDir>
#include <QFileInfo>
#include <QEasingCurve>
#include <QVariantAnimation>
#include <QIcon>
#include <QSize>
#include <QSizePolicy>
#include <QComboBox>
#include <QPushButton>
#include <QProgressDialog>
#include <QTextStream>
#include <QLabel>
#include <QElapsedTimer>
#include <QStringConverter>
#include <QThread>
#include <QTransform>
#include <QTimer>
#include <QMetaObject>
#include <QDateTime>

#include <QtConcurrent/QtConcurrent>
#include <QFuture>
#include <QFutureWatcher>

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <vector>
#include <string>
#include <exception>


// ============================================================
// HELPER FUNCTIONS
// ============================================================

namespace
{
    // Batch animation settings. Set kBatchAnimationMs to 0 to
    // skip the animation completely (fastest batch).
    constexpr int kBatchAnimationMs = 1500;

    // Convert QImage to an independent OpenCV BGR image.
    cv::Mat qImageToBGR(const QImage& image)
    {
        if (image.isNull())
            return cv::Mat();

        QImage converted =
            image.convertToFormat(QImage::Format_RGB888);

        cv::Mat rgb(
            converted.height(),
            converted.width(),
            CV_8UC3,
            converted.bits(),
            static_cast<size_t>(converted.bytesPerLine())
        );

        cv::Mat bgr;

        cv::cvtColor(
            rgb,
            bgr,
            cv::COLOR_RGB2BGR
        );

        return bgr.clone();
    }

    // Convert OpenCV BGR image to an independent QImage.
    QImage bgrToQImage(const cv::Mat& image)
    {
        if (image.empty())
            return QImage();

        cv::Mat rgb;

        if (image.channels() == 3)
        {
            cv::cvtColor(
                image,
                rgb,
                cv::COLOR_BGR2RGB
            );
        }
        else if (image.channels() == 1)
        {
            cv::cvtColor(
                image,
                rgb,
                cv::COLOR_GRAY2RGB
            );
        }
        else
        {
            return QImage();
        }

        QImage result(
            rgb.data,
            rgb.cols,
            rgb.rows,
            static_cast<qsizetype>(rgb.step),
            QImage::Format_RGB888
        );

        return result.copy();
    }

    // Display an image inside the existing preview area.
    // The image must never control the UI dimensions.
    void displayImage(
        QLabel* label,
        const QImage& image
    )
    {
        if (!label || image.isNull())
            return;

        const QSize availableSize =
            label->contentsRect().size();

        if (availableSize.isEmpty())
            return;

        QPixmap pixmap =
            QPixmap::fromImage(image);

        pixmap = pixmap.scaled(
            availableSize,
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        );

        label->setPixmap(pixmap);
    }

    // Calculate the longest edge angle of a rotated rectangle.
    double getLongestEdgeAngle(
        const cv::RotatedRect& box
    )
    {
        cv::Point2f corners[4];

        box.points(corners);

        double longestLength = 0.0;
        double longestAngle = 0.0;

        for (int i = 0; i < 4; ++i)
        {
            const cv::Point2f edge =
                corners[(i + 1) % 4] - corners[i];

            const double length = std::hypot(
                static_cast<double>(edge.x),
                static_cast<double>(edge.y)
            );

            if (length > longestLength)
            {
                longestLength = length;

                longestAngle =
                    std::atan2(
                        static_cast<double>(edge.y),
                        static_cast<double>(edge.x)
                    ) * 180.0 / CV_PI;
            }
        }

        return longestLength < 1e-6
            ? 9999.0
            : longestAngle;
    }

    // Calculate the angle correction required to make
    // the longest component edge horizontal or vertical.
    double calculateRotationCorrection(
        const cv::RotatedRect& box
    )
    {
        const double longestAngle =
            getLongestEdgeAngle(box);

        if (longestAngle == 9999.0)
            return 0.0;

        double normalizedAngle =
            std::fmod(longestAngle + 90.0, 180.0);

        if (normalizedAngle < 0.0)
            normalizedAngle += 180.0;

        normalizedAngle -= 90.0;

        const double targetAngle =
            std::round(normalizedAngle / 90.0) * 90.0;

        double correction =
            normalizedAngle - targetAngle;

        while (correction > 90.0)
            correction -= 180.0;

        while (correction < -90.0)
            correction += 180.0;

        if (std::abs(correction) < 0.15)
            correction = 0.0;

        return correction;
    }

    // Draw a rotated rectangle.
    void drawRotatedBox(
        cv::Mat& image,
        const cv::RotatedRect& box,
        const cv::Scalar& colour,
        int thickness = 3
    )
    {
        cv::Point2f corners[4];

        box.points(corners);

        for (int i = 0; i < 4; ++i)
        {
            cv::line(
                image,
                corners[i],
                corners[(i + 1) % 4],
                colour,
                thickness,
                cv::LINE_AA
            );
        }
    }
}


// ============================================================
// CONSTRUCTOR
// ============================================================

AutoCentering::AutoCentering(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    // IMAGE PREVIEW LAYOUT
    ui.originalImageLabel->setSizePolicy(
        QSizePolicy::Ignored,
        QSizePolicy::Expanding
    );

    ui.originalImageLabel->setMinimumSize(0, 0);
    ui.originalImageLabel->setMaximumSize(
        QWIDGETSIZE_MAX,
        QWIDGETSIZE_MAX
    );
    ui.originalImageLabel->setScaledContents(false);
    ui.originalImageLabel->setAlignment(Qt::AlignCenter);

    ui.lblimagetitle->setSizePolicy(
        QSizePolicy::Preferred,
        QSizePolicy::Fixed
    );
    ui.lblimagetitle->setFixedHeight(32);
    ui.lblimagetitle->setAlignment(
        Qt::AlignLeft | Qt::AlignVCenter
    );

    // LOAD DARK THEME
    QFile styleFile(":/styles/dark_minimal.qss");

    if (styleFile.open(QFile::ReadOnly | QFile::Text))
    {
        qApp->setStyleSheet(
            QString::fromUtf8(styleFile.readAll())
        );

        styleFile.close();
        qDebug() << "Dark Minimal Theme Loaded Successfully";
    }

    // LOAD YOLO MODEL
    detector = new YOLODetector(L"models/best.onnx");

    // CREATE BACKGROUND WORK WATCHERS
    openCVWatcher =
        new QFutureWatcher<OpenCVOnlySegmentationResult>(this);

    hybridWatcher =
        new QFutureWatcher<YOLOGuidedRefinementResult>(this);

    // ========================================================
    // OPENCV-ONLY COMPLETION
    // ========================================================

    connect(
        openCVWatcher,
        &QFutureWatcher<OpenCVOnlySegmentationResult>::finished,
        this,
        [this]()
        {
            OpenCVOnlySegmentationResult result =
                openCVWatcher->result();

            setProcessingState(false);

            if (!result.success || !result.outerContourFound)
            {
                ui.lblstatus->setText(
                    "Status : Component Not Found"
                );

                ui.lblImageStatus->setText(
                    QString::fromStdString(result.message)
                );

                QMessageBox::warning(
                    this,
                    "OpenCV Detection Failed",
                    QString::fromStdString(result.message)
                );

                return;
            }

            if (result.candidateVisualization.empty())
            {
                ui.lblstatus->setText("Status : Display Error");

                QMessageBox::warning(
                    this,
                    "Display Error",
                    "The OpenCV result image is empty."
                );

                return;
            }

            const cv::RotatedRect box =
                result.outerRotatedBoundingBox;

            if (box.size.width <= 0.0f ||
                box.size.height <= 0.0f)
            {
                QMessageBox::warning(
                    this,
                    "Invalid Bounding Box",
                    "OpenCV returned an invalid bounding box."
                );

                return;
            }

            displayImage(
                ui.originalImageLabel,
                bgrToQImage(result.candidateVisualization)
            );

            const double imageCenterX =
                result.candidateVisualization.cols / 2.0;

            const double imageCenterY =
                result.candidateVisualization.rows / 2.0;

            const double offsetX =
                box.center.x - imageCenterX;

            const double offsetY =
                box.center.y - imageCenterY;

            ui.labeloffsetx->setText(
                QString("Component Offset X : %1 px")
                .arg(offsetX, 0, 'f', 1)
            );

            ui.labeloffsety->setText(
                QString("Component Offset Y : %1 px")
                .arg(offsetY, 0, 'f', 1)
            );

            ui.labelconfidence->setText(
                "Confidence : N/A (OpenCV)"
            );

            ui.lblstatus->setText(
                "Status : Component Detected"
            );

            ui.lblImageStatus->setText(
                QString(
                    "Geometric Refinement successful.\n"
                    "Contour points: %1\n"
                    "Rotated width: %2 px\n"
                    "Rotated height: %3 px\n"
                    "Rotation angle: %4 degrees\n"
                    "Component centre: (%5, %6)\n"
                    "Offset X: %7 px\n"
                    "Offset Y: %8 px\n"
                    "Saved: output/open_cv_diagnostics/"
                    "opencv_main_component_result.png"
                )
                .arg(result.outerCableContour.size())
                .arg(box.size.width, 0, 'f', 1)
                .arg(box.size.height, 0, 'f', 1)
                .arg(box.angle, 0, 'f', 2)
                .arg(box.center.x, 0, 'f', 1)
                .arg(box.center.y, 0, 'f', 1)
                .arg(offsetX, 0, 'f', 1)
                .arg(offsetY, 0, 'f', 1)
            );
        }
    );

    // ========================================================
    // HYBRID COMPLETION
    // ========================================================

    connect(
        hybridWatcher,
        &QFutureWatcher<YOLOGuidedRefinementResult>::finished,
        this,
        [this]()
        {
            YOLOGuidedRefinementResult result =
                hybridWatcher->result();

            setProcessingState(false);

            if (!result.success || !result.refinedContourFound)
            {
                ui.lblstatus->setText(
                    "Status : Geometric Refinement Failed"
                );

                ui.lblImageStatus->setText(
                    QString::fromStdString(result.message)
                );

                QMessageBox::warning(
                    this,
                    "Hybrid Detection Failed",
                    QString::fromStdString(result.message)
                );

                return;
            }

            const cv::RotatedRect refinedBox =
                result.refinedRotatedBoundingBox;

            if (refinedBox.size.width <= 0.0f ||
                refinedBox.size.height <= 0.0f ||
                result.fullImageVisualization.empty())
            {
                ui.lblstatus->setText(
                    "Status : Invalid Hybrid Result"
                );

                QMessageBox::warning(
                    this,
                    "Hybrid Result Error",
                    "The refined bounding box or visualization is invalid."
                );

                return;
            }

            displayImage(
                ui.originalImageLabel,
                bgrToQImage(result.fullImageVisualization)
            );

            const double imageCenterX =
                result.fullImageVisualization.cols / 2.0;

            const double imageCenterY =
                result.fullImageVisualization.rows / 2.0;

            const double offsetX =
                refinedBox.center.x - imageCenterX;

            const double offsetY =
                refinedBox.center.y - imageCenterY;

            ui.labeloffsetx->setText(
                QString("Center Offset X : %1 px")
                .arg(offsetX, 0, 'f', 1)
            );

            ui.labeloffsety->setText(
                QString("Center Offset Y : %1 px")
                .arg(offsetY, 0, 'f', 1)
            );

            const float yoloConfidence =
                lastYoloConfidence.load();

            ui.labelconfidence->setText(
                yoloConfidence >= 0.0f
                ? QString("YOLO Confidence : %1 %")
                .arg(yoloConfidence * 100.0f, 0, 'f', 2)
                : QString("YOLO Confidence : N/A")
            );

            ui.lblstatus->setText(
                "Status : YOLO + Geometric Refinement Completed"
            );

            ui.lblImageStatus->setText(
                QString(
                    "YOLO + Geometric Refinement Successful.\n\n"
                    "ROI:\n"
                    "X: %1 px\n"
                    "Y: %2 px\n"
                    "Width: %3 px\n"
                    "Height: %4 px\n\n"
                    "Refined Component:\n"
                    "Contour Points: %5\n"
                    "Width: %6 px\n"
                    "Height: %7 px\n"
                    "Angle: %8 degrees\n\n"
                    "Refined Centre:\n"
                    "X: %9 px\n"
                    "Y: %10 px\n\n"
                    "Centre Offset:\n"
                    "X: %11 px\n"
                    "Y: %12 px\n\n"
                    "Saved: output/yolo_geometric_diagnostics/"
                    "yolo_geometric_refinement_result.png"
                )
                .arg(result.roi.x)
                .arg(result.roi.y)
                .arg(result.roi.width)
                .arg(result.roi.height)
                .arg(result.refinedContour.size())
                .arg(refinedBox.size.width, 0, 'f', 1)
                .arg(refinedBox.size.height, 0, 'f', 1)
                .arg(refinedBox.angle, 0, 'f', 2)
                .arg(refinedBox.center.x, 0, 'f', 1)
                .arg(refinedBox.center.y, 0, 'f', 1)
                .arg(offsetX, 0, 'f', 1)
                .arg(offsetY, 0, 'f', 1)
            );
        }
    );

    // CONNECT BUTTONS
    connect(ui.btnLoadImage, &QPushButton::clicked,
        this, &AutoCentering::loadImage);

    connect(ui.btnProcess, &QPushButton::clicked,
        this, &AutoCentering::processImage);

    connect(ui.btn_OpenCVOnly, &QPushButton::clicked,
        this, &AutoCentering::processOpenCVOnly);

    connect(ui.btnAutoCenter, &QPushButton::clicked,
        this, &AutoCentering::autoCenterImage);

    connect(ui.btn_yologf, &QPushButton::clicked,
        this, &AutoCentering::processYOLOGeometricRefinement);

    // "Save Result" button. It is found by its text, so this works
    // whatever its objectName is in the .ui file.
    const QList<QPushButton*> allButtons =
        findChildren<QPushButton*>();

    for (QPushButton* button : allButtons)
    {
        if (button->text().contains(
            "Save Result",
            Qt::CaseInsensitive))
        {
            btnSaveResult = button;
            break;
        }
    }

    if (btnSaveResult)
    {
        connect(
            btnSaveResult,
            &QPushButton::clicked,
            this,
            &AutoCentering::saveResult
        );
    }
    else
    {
        qDebug() << "Save Result button not found.";
    }

    setupBatchAutoMode();
}


// ============================================================
// DESTRUCTOR
// ============================================================

AutoCentering::~AutoCentering()
{
    if (shiftAnimation)
        shiftAnimation->stop();

    // Ask the batch worker to stop, then wait for it
    // BEFORE deleting the detector it may be using.
    batchCancelRequested = true;

    if (batchWatcher && batchWatcher->isRunning())
        batchWatcher->waitForFinished();

    if (openCVWatcher && openCVWatcher->isRunning())
        openCVWatcher->waitForFinished();

    if (hybridWatcher && hybridWatcher->isRunning())
        hybridWatcher->waitForFinished();

    delete detector;
    detector = nullptr;
}


// ============================================================
// LOAD IMAGE
// ============================================================

void AutoCentering::loadImage()
{
    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Open X-Ray Image",
        "",
        "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
    );

    if (fileName.isEmpty())
        return;

    QImage loadedImage(fileName);

    if (loadedImage.isNull())
    {
        QMessageBox::critical(
            this,
            "Image Error",
            "Failed to load the selected image."
        );

        return;
    }

    // Stop an existing animation.
    if (shiftAnimation)
    {
        shiftAnimation->stop();
        shiftAnimation->deleteLater();
        shiftAnimation = nullptr;
    }

    currentImagePath = fileName;
    currentImage = loadedImage;

    // The previous (unsaved) result no longer matches this image.
    pendingCorrectedImage.release();
    pendingPreviewImage.release();
    pendingBaseName.clear();

    // Reset detection method.
    activeMethod = DetectionMethod::None;

    displayImage(
        ui.originalImageLabel,
        currentImage
    );

    ui.lblImageStatus->setText("Image Loaded Successfully");
    ui.lblstatus->setText("Status : Waiting");
    ui.labeloffsetx->setText("Center Offset X : 0 px");
    ui.labeloffsety->setText("Center Offset Y : 0 px");
    ui.labelconfidence->setText("Confidence : --");
}


// ============================================================
// YOLO DETECTION
// ============================================================

void AutoCentering::processImage()
{
    if (currentImage.isNull())
    {
        QMessageBox::warning(
            this,
            "No Image",
            "Please load an X-ray image first."
        );

        return;
    }

    activeMethod = DetectionMethod::YOLO;

    if (!detector || !detector->isLoaded())
    {
        ui.lblstatus->setText("Status : YOLO Model Error");

        QMessageBox::critical(
            this,
            "YOLO Error",
            "The ONNX model is not loaded."
        );

        return;
    }

    try
    {
        cv::Mat bgr = qImageToBGR(currentImage);

        if (bgr.empty())
        {
            QMessageBox::warning(
                this,
                "Image Error",
                "Could not convert the image."
            );

            return;
        }

        const CableDetection detection =
            detector->detect(bgr);

        if (!detection.found)
        {
            ui.lblstatus->setText("Status : No Detection");

            ui.lblImageStatus->setText(
                "YOLO could not detect the component."
            );

            QMessageBox::information(
                this,
                "YOLO Detection",
                "No detection passed the confidence threshold."
            );

            return;
        }

        const float angleDegrees =
            detection.angleRadians * 180.0f /
            static_cast<float>(CV_PI);

        const cv::RotatedRect box(
            detection.center,
            detection.size,
            angleDegrees
        );

        cv::Mat preview = bgr.clone();

        drawRotatedBox(
            preview,
            box,
            cv::Scalar(0, 255, 0),
            3
        );

        cv::circle(
            preview,
            detection.center,
            6,
            cv::Scalar(0, 0, 255),
            cv::FILLED,
            cv::LINE_AA
        );

        displayImage(
            ui.originalImageLabel,
            bgrToQImage(preview)
        );

        const double imageCenterX = bgr.cols / 2.0;
        const double imageCenterY = bgr.rows / 2.0;

        const double offsetX =
            detection.center.x - imageCenterX;

        const double offsetY =
            detection.center.y - imageCenterY;

        ui.labeloffsetx->setText(
            QString("Center Offset X : %1 px")
            .arg(offsetX, 0, 'f', 1)
        );

        ui.labeloffsety->setText(
            QString("Center Offset Y : %1 px")
            .arg(offsetY, 0, 'f', 1)
        );

        ui.labelconfidence->setText(
            QString("Confidence : %1 %")
            .arg(detection.confidence * 100.0f, 0, 'f', 2)
        );

        ui.lblstatus->setText(
            "Status : YOLO Detection Completed"
        );

        ui.lblImageStatus->setText(
            QString(
                "YOLO component detection successful.\n"
                "Component centre: (%1, %2)\n"
                "Image centre: (%3, %4)\n"
                "Offset X: %5 px\n"
                "Offset Y: %6 px\n"
                "Rotation angle: %7 degrees"
            )
            .arg(detection.center.x, 0, 'f', 1)
            .arg(detection.center.y, 0, 'f', 1)
            .arg(imageCenterX, 0, 'f', 1)
            .arg(imageCenterY, 0, 'f', 1)
            .arg(offsetX, 0, 'f', 1)
            .arg(offsetY, 0, 'f', 1)
            .arg(angleDegrees, 0, 'f', 2)
        );
    }
    catch (const cv::Exception& exception)
    {
        ui.lblstatus->setText("Status : YOLO Error");

        QMessageBox::critical(
            this,
            "OpenCV Error",
            QString::fromLocal8Bit(exception.what())
        );
    }
    catch (const std::exception& exception)
    {
        ui.lblstatus->setText("Status : YOLO Error");

        QMessageBox::critical(
            this,
            "Detection Error",
            QString::fromLocal8Bit(exception.what())
        );
    }
}


// ============================================================
// OPENCV-ONLY PROCESSING (RUNS ON A WORKER THREAD)
// ============================================================

void AutoCentering::processOpenCVOnly()
{
    if (isProcessing)
        return;

    if (currentImage.isNull())
    {
        QMessageBox::warning(
            this,
            "No Image",
            "Please load an X-ray image first."
        );

        return;
    }

    activeMethod = DetectionMethod::GeometricRefinement;

    cv::Mat bgr = qImageToBGR(currentImage);

    if (bgr.empty())
    {
        QMessageBox::warning(
            this,
            "Image Error",
            "Could not convert the image."
        );

        return;
    }

    const QString outputFolder =
        QDir::currentPath() +
        "/output/open_cv_diagnostics";

    if (!QDir().mkpath(outputFolder))
    {
        QMessageBox::warning(
            this,
            "Output Error",
            "Could not create the diagnostic folder."
        );

        return;
    }

    setProcessingState(true);

    ui.lblstatus->setText(
        "Status : Geometric Refinement Running..."
    );

    ui.lblImageStatus->setText(
        "Detecting the outer boundary of the main component..."
    );

    const std::string outputFolderStd =
        outputFolder.toStdString();

    // NOTE: never touch ui.* inside this lambda.
    openCVWatcher->setFuture(
        QtConcurrent::run(
            [bgr, outputFolderStd]() -> OpenCVOnlySegmentationResult
            {
                try
                {
                    OpenCVOnlySegmentationResult result =
                        OpenCVOnlyGeometry::segmentFullImage(
                            bgr,
                            outputFolderStd
                        );

                    if (result.success &&
                        result.outerContourFound &&
                        !result.candidateVisualization.empty())
                    {
                        cv::imwrite(
                            outputFolderStd +
                            "/opencv_main_component_result.png",
                            result.candidateVisualization
                        );
                    }

                    return result;
                }
                catch (const std::exception& e)
                {
                    OpenCVOnlySegmentationResult failed;
                    failed.success = false;
                    failed.outerContourFound = false;
                    failed.message =
                        std::string("Processing error: ") + e.what();
                    return failed;
                }
                catch (...)
                {
                    OpenCVOnlySegmentationResult failed;
                    failed.success = false;
                    failed.outerContourFound = false;
                    failed.message = "Unknown error occurred.";
                    return failed;
                }
            }
        )
    );
}


// ============================================================
// YOLO + GEOMETRIC REFINEMENT (RUNS ON A WORKER THREAD)
// ============================================================

void AutoCentering::processYOLOGeometricRefinement()
{
    if (isProcessing)
        return;

    if (currentImage.isNull())
    {
        QMessageBox::warning(
            this,
            "No Image",
            "Please load an X-ray image first."
        );

        return;
    }

    if (!detector || !detector->isLoaded())
    {
        ui.lblstatus->setText("Status : YOLO Model Error");

        QMessageBox::critical(
            this,
            "YOLO Error",
            "The ONNX model is not loaded."
        );

        return;
    }

    activeMethod = DetectionMethod::YOLOGeometricRefinement;

    cv::Mat bgr = qImageToBGR(currentImage);

    if (bgr.empty())
    {
        QMessageBox::warning(
            this,
            "Image Error",
            "Could not convert the image."
        );

        return;
    }

    const QString outputFolder =
        QDir::currentPath() +
        "/output/yolo_geometric_diagnostics";

    if (!QDir().mkpath(outputFolder))
    {
        QMessageBox::warning(
            this,
            "Output Error",
            "Could not create the hybrid diagnostic folder."
        );

        return;
    }

    setProcessingState(true);

    lastYoloConfidence = -1.0f;

    ui.lblstatus->setText(
        "Status : YOLO + Geometric Refinement Running..."
    );

    const std::string outputFolderStd =
        outputFolder.toStdString();

    YOLODetector* yoloDetector = detector;

    // NOTE: never touch ui.* inside this lambda.
    hybridWatcher->setFuture(
        QtConcurrent::run(
            [this, yoloDetector, bgr, outputFolderStd]()
            -> YOLOGuidedRefinementResult
            {
                YOLOGuidedRefinementResult failed;
                failed.success = false;
                failed.refinedContourFound = false;

                try
                {
                    QElapsedTimer totalTimer;
                    totalTimer.start();

                    const CableDetection detection =
                        yoloDetector->detect(bgr);

                    qDebug() << "[PERFORMANCE] YOLO Detection:"
                        << totalTimer.elapsed() << "ms";

                    if (!detection.found)
                    {
                        failed.message =
                            "YOLO could not detect the component.";

                        return failed;
                    }

                    lastYoloConfidence = detection.confidence;

                    const float angleDegrees =
                        detection.angleRadians * 180.0f /
                        static_cast<float>(CV_PI);

                    const cv::RotatedRect yoloBox(
                        detection.center,
                        detection.size,
                        angleDegrees
                    );

                    QElapsedTimer refinementTimer;
                    refinementTimer.start();

                    YOLOGuidedRefinementResult result =
                        OpenCVOnlyGeometry::refineWithinYOLORegion(
                            bgr,
                            yoloBox,
                            outputFolderStd,
                            0.25f
                        );

                    qDebug() << "[PERFORMANCE] OpenCV Refinement:"
                        << refinementTimer.elapsed() << "ms";

                    if (result.success &&
                        result.refinedContourFound &&
                        !result.fullImageVisualization.empty())
                    {
                        cv::imwrite(
                            outputFolderStd +
                            "/yolo_geometric_refinement_result.png",
                            result.fullImageVisualization
                        );
                    }

                    return result;
                }
                catch (const std::exception& e)
                {
                    failed.message =
                        std::string("Processing error: ") + e.what();

                    return failed;
                }
                catch (...)
                {
                    failed.message = "Unknown error occurred.";

                    return failed;
                }
            }
        )
    );
}


// ============================================================
// AUTO CENTERING (SINGLE IMAGE)
// ============================================================

void AutoCentering::autoCenterImage()
{
    struct DetectionInfo
    {
        bool found = false;
        cv::Point2f center;
        cv::RotatedRect box;
        float confidence = -1.0f;
        QString message;
    };

    auto setCenteringFailed = [this]()
        {
            ui.lblstatus->setText("Status : Failed");
            ui.labeloffsetx->setText("Center Offset X : --");
            ui.labeloffsety->setText("Center Offset Y : --");
            ui.labelconfidence->setText("Confidence : --");
        };

    if (currentImage.isNull())
    {
        setCenteringFailed();

        QMessageBox::warning(
            this,
            "Auto Center",
            "Please load an X-ray image first."
        );

        return;
    }

    if (activeMethod == DetectionMethod::None)
    {
        setCenteringFailed();

        QMessageBox::warning(
            this,
            "Auto Center",
            "Please select a detection method first."
        );

        return;
    }

    // A new run invalidates the previous (unsaved) result.
    pendingCorrectedImage.release();
    pendingPreviewImage.release();
    pendingBaseName.clear();

    const DetectionMethod selectedMethod = activeMethod;

    QString methodName;

    switch (selectedMethod)
    {
    case DetectionMethod::YOLO:
        methodName = "YOLO";
        break;
    case DetectionMethod::GeometricRefinement:
        methodName = "Geometric Refinement";
        break;
    case DetectionMethod::YOLOGeometricRefinement:
        methodName = "YOLO + Geometric Refinement";
        break;
    default:
        methodName = "Unknown";
        break;
    }

    if (selectedMethod == DetectionMethod::YOLO ||
        selectedMethod == DetectionMethod::YOLOGeometricRefinement)
    {
        if (!detector || !detector->isLoaded())
        {
            setCenteringFailed();

            QMessageBox::critical(
                this,
                "YOLO Error",
                "The ONNX model is not loaded."
            );

            return;
        }
    }

    if (shiftAnimation)
    {
        shiftAnimation->stop();
        shiftAnimation->deleteLater();
        shiftAnimation = nullptr;
    }

    ui.lblstatus->setText("Status : Detecting Component...");
    ui.labeloffsetx->setText("Center Offset X : Processing...");
    ui.labeloffsety->setText("Center Offset Y : Processing...");
    ui.labelconfidence->setText("Confidence : Processing...");

    cv::Mat original = qImageToBGR(currentImage);

    if (original.empty())
    {
        setCenteringFailed();

        QMessageBox::warning(
            this,
            "Image Error",
            "Could not convert the original image."
        );

        return;
    }

    const int originalWidth = original.cols;
    const int originalHeight = original.rows;

    const cv::Point2f outputCenter(
        originalWidth / 2.0f,
        originalHeight / 2.0f
    );

    std::string diagnosticDirectory;

    if (selectedMethod == DetectionMethod::GeometricRefinement ||
        selectedMethod == DetectionMethod::YOLOGeometricRefinement)
    {
        const QString diagnosticFolder =
            QDir::currentPath() +
            "/output/auto_center_geometry_diagnostics";

        if (!QDir().mkpath(diagnosticFolder))
        {
            setCenteringFailed();

            QMessageBox::warning(
                this,
                "Auto Center",
                "Could not create the diagnostic folder."
            );

            return;
        }

        diagnosticDirectory = diagnosticFolder.toStdString();
    }

    // Common detection function for all three methods.
    auto detectObject =
        [this, selectedMethod](
            const cv::Mat& inputImage,
            const std::string& diagnosticPath
            ) -> DetectionInfo
        {
            DetectionInfo info;

            if (inputImage.empty())
            {
                info.message = "Input image is empty.";
                return info;
            }

            // METHOD 1: YOLO
            if (selectedMethod == DetectionMethod::YOLO)
            {
                if (!detector || !detector->isLoaded())
                {
                    info.message = "YOLO model is not loaded.";
                    return info;
                }

                const CableDetection detection =
                    detector->detect(inputImage);

                if (!detection.found)
                {
                    info.message =
                        "YOLO could not detect the component.";
                    return info;
                }

                const float angleDegrees =
                    detection.angleRadians * 180.0f /
                    static_cast<float>(CV_PI);

                info.center = detection.center;
                info.box = cv::RotatedRect(
                    detection.center,
                    detection.size,
                    angleDegrees
                );
                info.confidence = detection.confidence;
                info.found = true;
                info.message = "YOLO detection successful.";
                return info;
            }

            // METHOD 2: GEOMETRIC REFINEMENT
            if (selectedMethod == DetectionMethod::GeometricRefinement)
            {
                OpenCVOnlySegmentationResult result =
                    OpenCVOnlyGeometry::segmentFullImage(
                        inputImage,
                        diagnosticPath
                    );

                if (!result.success || !result.outerContourFound)
                {
                    info.message =
                        QString::fromStdString(result.message);
                    return info;
                }

                const cv::RotatedRect box =
                    result.outerRotatedBoundingBox;

                if (box.size.width <= 0.0f ||
                    box.size.height <= 0.0f)
                {
                    info.message =
                        "Geometric Refinement returned an invalid bounding box.";
                    return info;
                }

                info.center = box.center;
                info.box = box;
                info.confidence = -1.0f;
                info.found = true;
                info.message = "Geometric Refinement successful.";
                return info;
            }

            // METHOD 3: YOLO + GEOMETRIC REFINEMENT
            if (selectedMethod ==
                DetectionMethod::YOLOGeometricRefinement)
            {
                if (!detector || !detector->isLoaded())
                {
                    info.message = "YOLO model is not loaded.";
                    return info;
                }

                const CableDetection detection =
                    detector->detect(inputImage);

                if (!detection.found)
                {
                    info.message =
                        "YOLO could not detect the component.";
                    return info;
                }

                const float angleDegrees =
                    detection.angleRadians * 180.0f /
                    static_cast<float>(CV_PI);

                const cv::RotatedRect yoloBox(
                    detection.center,
                    detection.size,
                    angleDegrees
                );

                YOLOGuidedRefinementResult result =
                    OpenCVOnlyGeometry::refineWithinYOLORegion(
                        inputImage,
                        yoloBox,
                        diagnosticPath,
                        0.25f
                    );

                if (!result.success || !result.refinedContourFound)
                {
                    info.message =
                        QString::fromStdString(result.message);
                    return info;
                }

                const cv::RotatedRect refinedBox =
                    result.refinedRotatedBoundingBox;

                if (refinedBox.size.width <= 0.0f ||
                    refinedBox.size.height <= 0.0f)
                {
                    info.message = "Invalid refined bounding box.";
                    return info;
                }

                info.center = refinedBox.center;
                info.box = refinedBox;
                info.confidence = detection.confidence;
                info.found = true;
                info.message =
                    "YOLO + Geometric Refinement successful.";
                return info;
            }

            info.message = "No detection method selected.";
            return info;
        };

    // INITIAL DETECTION
    DetectionInfo initialDetection;

    try
    {
        initialDetection = detectObject(original, diagnosticDirectory);
    }
    catch (const std::exception& exception)
    {
        setCenteringFailed();

        QMessageBox::critical(
            this,
            "Detection Error",
            QString::fromLocal8Bit(exception.what())
        );

        return;
    }

    if (!initialDetection.found)
    {
        setCenteringFailed();

        ui.lblImageStatus->setText(initialDetection.message);

        QMessageBox::information(
            this,
            "Auto Center",
            QString(
                "%1 detection failed.\n\n%2\n\n"
                "The original image remains unchanged."
            )
            .arg(methodName)
            .arg(initialDetection.message)
        );

        return;
    }

    // ROTATION CORRECTION
    const double correctionAngle =
        calculateRotationCorrection(initialDetection.box);

    // ROTATE IMAGE WITHOUT CROPPING
    const cv::Point2f originalCenter(
        original.cols / 2.0f,
        original.rows / 2.0f
    );

    cv::Mat rotationMatrix =
        cv::getRotationMatrix2D(originalCenter, correctionAngle, 1.0);

    const double cosine =
        std::abs(rotationMatrix.at<double>(0, 0));

    const double sine =
        std::abs(rotationMatrix.at<double>(0, 1));

    const int rotatedWidth = std::max(
        1,
        static_cast<int>(
            std::ceil(original.rows * sine + original.cols * cosine))
    );

    const int rotatedHeight = std::max(
        1,
        static_cast<int>(
            std::ceil(original.rows * cosine + original.cols * sine))
    );

    rotationMatrix.at<double>(0, 2) +=
        rotatedWidth / 2.0 - originalCenter.x;

    rotationMatrix.at<double>(1, 2) +=
        rotatedHeight / 2.0 - originalCenter.y;

    cv::Mat rotated;

    try
    {
        cv::warpAffine(
            original,
            rotated,
            rotationMatrix,
            cv::Size(rotatedWidth, rotatedHeight),
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(255, 255, 255)
        );
    }
    catch (const cv::Exception& exception)
    {
        setCenteringFailed();

        QMessageBox::critical(
            this,
            "Rotation Error",
            QString::fromLocal8Bit(exception.what())
        );

        return;
    }

    // DETECT AFTER ROTATION
    DetectionInfo rotatedDetection;

    try
    {
        rotatedDetection = detectObject(rotated, diagnosticDirectory);
    }
    catch (const std::exception& exception)
    {
        setCenteringFailed();

        QMessageBox::critical(
            this,
            "Detection Error",
            QString::fromLocal8Bit(exception.what())
        );

        return;
    }

    if (!rotatedDetection.found)
    {
        setCenteringFailed();

        QMessageBox::warning(
            this,
            "Auto Center",
            QString(
                "%1 could not detect the component after rotation.\n\n%2"
            )
            .arg(methodName)
            .arg(rotatedDetection.message)
        );

        return;
    }

    // TRANSLATION
    const double shiftX =
        outputCenter.x - rotatedDetection.center.x;

    const double shiftY =
        outputCenter.y - rotatedDetection.center.y;

    cv::Mat finalTransform = rotationMatrix.clone();

    finalTransform.at<double>(0, 2) += shiftX;
    finalTransform.at<double>(1, 2) += shiftY;

    // CORRECTED IMAGE
    cv::Mat corrected;

    try
    {
        cv::warpAffine(
            original,
            corrected,
            finalTransform,
            cv::Size(originalWidth, originalHeight),
            cv::INTER_CUBIC,
            cv::BORDER_CONSTANT,
            cv::Scalar(255, 255, 255)
        );
    }
    catch (const cv::Exception& exception)
    {
        setCenteringFailed();

        QMessageBox::critical(
            this,
            "Translation Error",
            QString::fromLocal8Bit(exception.what())
        );

        return;
    }

    // FINAL DETECTION
    DetectionInfo finalDetection;

    try
    {
        finalDetection = detectObject(corrected, diagnosticDirectory);
    }
    catch (const std::exception& exception)
    {
        setCenteringFailed();

        QMessageBox::critical(
            this,
            "Final Detection Error",
            QString::fromLocal8Bit(exception.what())
        );

        return;
    }

    if (!finalDetection.found)
    {
        setCenteringFailed();

        QMessageBox::warning(
            this,
            "Final Detection Failed",
            QString(
                "The corrected image was created, but final detection failed.\n\n%1"
            )
            .arg(finalDetection.message)
        );

        return;
    }

    // FINAL PREVIEW
    cv::Mat preview = corrected.clone();

    drawRotatedBox(
        preview,
        finalDetection.box,
        cv::Scalar(0, 255, 0),
        3
    );

    cv::circle(
        preview,
        finalDetection.center,
        6,
        cv::Scalar(0, 0, 255),
        cv::FILLED,
        cv::LINE_AA
    );

    cv::drawMarker(
        preview,
        cv::Point(originalWidth / 2, originalHeight / 2),
        cv::Scalar(255, 0, 0),
        cv::MARKER_CROSS,
        25,
        2,
        cv::LINE_AA
    );

    // NOTE: nothing is written to disk here. The corrected image and
    // the preview are kept in memory, and are only saved when the
    // user clicks "Save Result".
    QString baseName =
        QFileInfo(currentImagePath).completeBaseName();

    if (baseName.isEmpty())
        baseName = "xray_image";

    QString methodSuffix;

    switch (selectedMethod)
    {
    case DetectionMethod::YOLO:
        methodSuffix = "_yolo";
        break;
    case DetectionMethod::GeometricRefinement:
        methodSuffix = "_geometric";
        break;
    case DetectionMethod::YOLOGeometricRefinement:
        methodSuffix = "_yolo_geometric";
        break;
    default:
        methodSuffix = "_unknown";
        break;
    }

    pendingCorrectedImage = corrected.clone();
    pendingPreviewImage = preview.clone();
    pendingBaseName = baseName + methodSuffix;

    const QImage finalDisplayImage = bgrToQImage(preview);

    if (finalDisplayImage.isNull())
    {
        setCenteringFailed();

        QMessageBox::warning(
            this,
            "Display Error",
            "Could not prepare the final preview."
        );

        return;
    }

    // ANIMATION DATA
    const cv::Mat animationImage = original.clone();
    const cv::RotatedRect animationBox = initialDetection.box;
    const cv::Point2f animationCenter = initialDetection.center;
    const cv::Mat animationTransform = finalTransform.clone();

    // UPDATE UI
    if (initialDetection.confidence >= 0.0f)
    {
        ui.labelconfidence->setText(
            QString("Confidence : %1 %")
            .arg(initialDetection.confidence * 100.0f, 0, 'f', 2)
        );
    }
    else
    {
        ui.labelconfidence->setText("Confidence : N/A (OpenCV)");
    }

    ui.lblstatus->setText("Status : Starting Auto Center...");

    ui.lblImageStatus->setText(
        QString(
            "%1 Auto Center\n"
            "Rotation correction: %2 degrees\n"
            "Translation X: %3 px\n"
            "Translation Y: %4 px\n"
            "Preparing centering animation..."
        )
        .arg(methodName)
        .arg(correctionAngle, 0, 'f', 2)
        .arg(shiftX, 0, 'f', 1)
        .arg(shiftY, 0, 'f', 1)
    );

    // START ANIMATION
    shiftAnimation = new QVariantAnimation(this);

    shiftAnimation->setDuration(4000);
    shiftAnimation->setStartValue(0.0);
    shiftAnimation->setEndValue(1.0);
    shiftAnimation->setEasingCurve(QEasingCurve::InOutSine);

    connect(
        shiftAnimation,
        &QVariantAnimation::valueChanged,
        this,
        [
            this,
            animationImage,
            animationBox,
            animationCenter,
            animationTransform,
            outputCenter,
            originalWidth,
            originalHeight
        ](const QVariant& value)
        {
            const double progress = value.toDouble();

            // Interpolate affine transformation.
            cv::Mat currentTransform = cv::Mat::eye(2, 3, CV_64F);

            for (int row = 0; row < 2; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    const double startValue =
                        row == column ? 1.0 : 0.0;

                    const double finalValue =
                        animationTransform.at<double>(row, column);

                    currentTransform.at<double>(row, column) =
                        startValue +
                        (finalValue - startValue) * progress;
                }
            }

            cv::Mat frame;

            cv::warpAffine(
                animationImage,
                frame,
                currentTransform,
                cv::Size(originalWidth, originalHeight),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(255, 255, 255)
            );

            // Transform bounding box.
            cv::Point2f corners[4];

            animationBox.points(corners);

            for (int i = 0; i < 4; ++i)
            {
                const cv::Point2f point = corners[i];

                corners[i] = cv::Point2f(
                    static_cast<float>(
                        currentTransform.at<double>(0, 0) * point.x +
                        currentTransform.at<double>(0, 1) * point.y +
                        currentTransform.at<double>(0, 2)),
                    static_cast<float>(
                        currentTransform.at<double>(1, 0) * point.x +
                        currentTransform.at<double>(1, 1) * point.y +
                        currentTransform.at<double>(1, 2))
                );
            }

            for (int i = 0; i < 4; ++i)
            {
                cv::line(
                    frame,
                    corners[i],
                    corners[(i + 1) % 4],
                    cv::Scalar(0, 255, 0),
                    3,
                    cv::LINE_AA
                );
            }

            // Transform component centre.
            const cv::Point2f movingCenter(
                static_cast<float>(
                    currentTransform.at<double>(0, 0) * animationCenter.x +
                    currentTransform.at<double>(0, 1) * animationCenter.y +
                    currentTransform.at<double>(0, 2)),
                static_cast<float>(
                    currentTransform.at<double>(1, 0) * animationCenter.x +
                    currentTransform.at<double>(1, 1) * animationCenter.y +
                    currentTransform.at<double>(1, 2))
            );

            cv::circle(
                frame,
                movingCenter,
                6,
                cv::Scalar(0, 0, 255),
                cv::FILLED,
                cv::LINE_AA
            );

            // Target centre.
            cv::drawMarker(
                frame,
                cv::Point(originalWidth / 2, originalHeight / 2),
                cv::Scalar(255, 0, 0),
                cv::MARKER_CROSS,
                25,
                2,
                cv::LINE_AA
            );

            displayImage(
                ui.originalImageLabel,
                bgrToQImage(frame)
            );

            const double remainingX = movingCenter.x - outputCenter.x;
            const double remainingY = movingCenter.y - outputCenter.y;

            ui.labeloffsetx->setText(
                QString("Center Offset X : %1 px")
                .arg(remainingX, 0, 'f', 1)
            );

            ui.labeloffsety->setText(
                QString("Center Offset Y : %1 px")
                .arg(remainingY, 0, 'f', 1)
            );

            if (progress < 1.0)
            {
                ui.lblstatus->setText(
                    QString("Status : Auto Centering %1%")
                    .arg(static_cast<int>(progress * 100.0))
                );
            }
        }
    );

    connect(
        shiftAnimation,
        &QVariantAnimation::finished,
        this,
        [
            this,
            finalDisplayImage,
            finalDetection,
            originalWidth,
            originalHeight,
            correctionAngle,
            methodName
        ]()
        {
            displayImage(ui.originalImageLabel, finalDisplayImage);

            const double finalOffsetX =
                finalDetection.center.x - originalWidth / 2.0;

            const double finalOffsetY =
                finalDetection.center.y - originalHeight / 2.0;

            ui.lblstatus->setText("Status : Completed");

            ui.labeloffsetx->setText(
                QString("Center Offset X : %1 px")
                .arg(finalOffsetX, 0, 'f', 2)
            );

            ui.labeloffsety->setText(
                QString("Center Offset Y : %1 px")
                .arg(finalOffsetY, 0, 'f', 2)
            );

            if (finalDetection.confidence >= 0.0f)
            {
                ui.labelconfidence->setText(
                    QString("Confidence : %1 %")
                    .arg(finalDetection.confidence * 100.0f, 0, 'f', 2)
                );
            }
            else
            {
                ui.labelconfidence->setText("Confidence : N/A (OpenCV)");
            }

            ui.lblImageStatus->setText(
                QString(
                    "%1 Auto Center completed.\n"
                    "Output dimensions: %2 x %3 px\n"
                    "Rotation correction: %4 degrees\n"
                    "Final offset X: %5 px\n"
                    "Final offset Y: %6 px\n\n"
                    "Not saved yet. Click Save Result to save."
                )
                .arg(methodName)
                .arg(originalWidth)
                .arg(originalHeight)
                .arg(correctionAngle, 0, 'f', 2)
                .arg(finalOffsetX, 0, 'f', 2)
                .arg(finalOffsetY, 0, 'f', 2)
            );

            if (shiftAnimation)
            {
                shiftAnimation->deleteLater();
                shiftAnimation = nullptr;
            }
        }
    );

    shiftAnimation->start();
}


// ============================================================
// BATCH AUTO MODE - SETUP
// ============================================================

void AutoCentering::setupBatchAutoMode()
{
    if (!ui.grpBatchAutoMode)
    {
        qDebug() << "Batch Auto Mode group box not found.";
        return;
    }

    cmbBatchMethod = ui.cmbBatchMethod;
    lblBatchFolder = ui.lblBatchFolder;
    btnBrowseBatchFolder = ui.btnBrowseBatchFolder;
    btnStartBatch = ui.btnStartBatch;

    if (!cmbBatchMethod ||
        !lblBatchFolder ||
        !btnBrowseBatchFolder ||
        !btnStartBatch)
    {
        qDebug() << "Batch Auto Mode UI widgets are missing.";
        return;
    }

    cmbBatchMethod->clear();

    cmbBatchMethod->addItem(
        "YOLO",
        static_cast<int>(DetectionMethod::YOLO)
    );

    cmbBatchMethod->addItem(
        "Geometric Refinement",
        static_cast<int>(DetectionMethod::GeometricRefinement)
    );

    cmbBatchMethod->addItem(
        "YOLO + Geometric Refinement",
        static_cast<int>(DetectionMethod::YOLOGeometricRefinement)
    );

    lblBatchFolder->setText("No folder selected");

    connect(
        btnBrowseBatchFolder,
        &QPushButton::clicked,
        this,
        &AutoCentering::browseBatchFolder
    );

    connect(
        btnStartBatch,
        &QPushButton::clicked,
        this,
        &AutoCentering::startBatchAutoMode
    );
}


// ============================================================
// BATCH AUTO MODE - BROWSE INPUT FOLDER
// ============================================================

void AutoCentering::browseBatchFolder()
{
    QString folderPath = QFileDialog::getExistingDirectory(
        this,
        "Select X-Ray Image Folder",
        QDir::currentPath(),
        QFileDialog::ShowDirsOnly
    );

    if (folderPath.isEmpty())
        return;

    QDir selectedFolder(folderPath);

    QStringList filters;
    filters << "*.png"
        << "*.jpg"
        << "*.jpeg"
        << "*.bmp"
        << "*.tif"
        << "*.tiff";

    QStringList imageFiles = selectedFolder.entryList(
        filters,
        QDir::Files,
        QDir::Name
    );

    if (imageFiles.isEmpty())
    {
        QMessageBox::warning(
            this,
            "No Images Found",
            "The selected folder does not contain supported image files."
        );

        return;
    }

    batchInputDirectory = folderPath;

    lblBatchFolder->setText(
        QString("%1 images found\n%2")
        .arg(imageFiles.size())
        .arg(folderPath)
    );

    qDebug() << "Batch folder selected:" << folderPath;
    qDebug() << "Number of images:" << imageFiles.size();
}


// ============================================================
// BATCH AUTO MODE - START
// ============================================================

void AutoCentering::startBatchAutoMode()
{
    // While a batch is running, this button acts as "Cancel Batch".
    if (batchWatcher && batchWatcher->isRunning())
    {
        batchCancelRequested = true;

        if (btnStartBatch)
        {
            btnStartBatch->setText("Cancelling...");
            btnStartBatch->setEnabled(false);
        }

        ui.lblstatus->setText(
            "Status : Cancelling after the current image..."
        );

        return;
    }

    if (isProcessing)
        return;

    // 1. CHECK INPUT FOLDER
    if (batchInputDirectory.isEmpty())
    {
        QMessageBox::warning(
            this,
            "Batch Auto Mode",
            "Please select an input folder first."
        );

        return;
    }

    QDir inputFolder(batchInputDirectory);

    if (!inputFolder.exists())
    {
        QMessageBox::warning(
            this,
            "Batch Auto Mode",
            "The selected input folder does not exist."
        );

        return;
    }

    // 2. CHECK DETECTION METHOD
    if (!cmbBatchMethod || cmbBatchMethod->currentIndex() < 0)
    {
        QMessageBox::warning(
            this,
            "Batch Auto Mode",
            "Please select a detection method."
        );

        return;
    }

    const DetectionMethod selectedMethod =
        static_cast<DetectionMethod>(
            cmbBatchMethod->currentData().toInt()
            );

    if (selectedMethod == DetectionMethod::None)
    {
        QMessageBox::warning(
            this,
            "Batch Auto Mode",
            "Please select a valid detection method."
        );

        return;
    }

    // 3. CHECK YOLO MODEL
    if (selectedMethod == DetectionMethod::YOLO ||
        selectedMethod == DetectionMethod::YOLOGeometricRefinement)
    {
        if (!detector || !detector->isLoaded())
        {
            QMessageBox::critical(
                this,
                "YOLO Model Error",
                "The YOLO ONNX model is not loaded."
            );

            return;
        }
    }

    // 4. FIND SUPPORTED IMAGES
    QStringList filters;

    filters << "*.png"
        << "*.jpg"
        << "*.jpeg"
        << "*.bmp"
        << "*.tif"
        << "*.tiff";

    const QStringList imageFiles =
        inputFolder.entryList(
            filters,
            QDir::Files,
            QDir::Name
        );

    if (imageFiles.isEmpty())
    {
        QMessageBox::warning(
            this,
            "Batch Auto Mode",
            "No supported images were found in the selected folder."
        );

        return;
    }

    // 5. FULL IMAGE PATHS
    QStringList imagePaths;

    for (const QString& fileName : imageFiles)
        imagePaths.append(inputFolder.filePath(fileName));

    // 6. OUTPUT DIRECTORY
    // A NEW folder for every run, so results are never mixed with,
    // or confused with, files from an earlier run.
    const QString runStamp =
        QDateTime::currentDateTime().toString("yyyyMMdd_HHmmss");

    const QString outputDirectory =
        QDir::currentPath() +
        "/output/batch_auto_centering/run_" + runStamp;

    if (!QDir().mkpath(outputDirectory))
    {
        QMessageBox::critical(
            this,
            "Output Error",
            "Could not create the batch output folder."
        );

        return;
    }

    // 7. START BATCH (returns immediately; runs in the background).
    // The buttons are re-enabled by setProcessingState(false)
    // in the batchWatcher finished handler.
    processBatchImages(
        imagePaths,
        outputDirectory,
        selectedMethod
    );
}


// ============================================================
// BATCH AUTO MODE - BACKGROUND BATCH CONTROLLER
// ============================================================
//
// The whole batch loop runs on ONE worker thread.
// The worker never touches ui.* or batchProgress directly.
// It sends progress to the GUI thread with queued invocations.
// ============================================================

void AutoCentering::processBatchImages(
    const QStringList& imagePaths,
    const QString& outputDirectory,
    DetectionMethod selectedMethod)
{
    if (isProcessing ||
        (batchWatcher && batchWatcher->isRunning()))
        return;

    // --------------------------------------------------------
    // 1. OUTPUT FOLDERS (GUI thread, cheap)
    // --------------------------------------------------------

    QDir outputDir(outputDirectory);

    const QString correctedDirectory =
        outputDir.filePath("corrected");

    const QString previewDirectory =
        outputDir.filePath("previews");

    const QString diagnosticDirectory =
        outputDir.filePath("diagnostics");

    const QString csvPath =
        outputDir.filePath("batch_results.csv");

    if (!QDir().mkpath(correctedDirectory) ||
        !QDir().mkpath(previewDirectory) ||
        !QDir().mkpath(diagnosticDirectory))
    {
        QMessageBox::critical(
            this,
            "Batch Processing Error",
            "Unable to create the output folders."
        );

        return;
    }

    // --------------------------------------------------------
    // 2. ONE-TIME WATCHER SETUP
    // --------------------------------------------------------

    if (!batchWatcher)
    {
        batchWatcher = new QFutureWatcher<BatchSummary>(this);

        connect(
            batchWatcher,
            &QFutureWatcher<BatchSummary>::finished,
            this,
            [this]()
            {
                const BatchSummary s = batchWatcher->result();

                setProcessingState(false);

                if (btnStartBatch)
                    btnStartBatch->setText("Start Batch Processing");

                if (!s.errorMessage.isEmpty())
                {
                    ui.lblstatus->setText("Status : Batch Failed");

                    QMessageBox::critical(
                        this,
                        "Batch Processing Error",
                        s.errorMessage
                    );

                    return;
                }

                ui.lblstatus->setText(
                    s.cancelled
                    ? "Status : Batch Cancelled"
                    : "Status : Batch Completed"
                );

                ui.lblImageStatus->setText(
                    QString("Results saved to:\n%1")
                    .arg(QDir::toNativeSeparators(s.outputDirectory))
                );

                QString summary;

                summary += "Batch processing completed.\n\n";
                summary += QString("Selected method: %1\n")
                    .arg(s.methodName);
                summary += QString("Total images: %1\n").arg(s.total);
                summary += QString("Processed: %1\n").arg(s.processed);
                summary += QString("Successful: %1\n").arg(s.success);
                summary += QString("Failed: %1\n").arg(s.failed);

                if (s.cancelled)
                {
                    summary += "Cancelled: Yes\n";
                    summary += QString("Not processed: %1\n")
                        .arg(s.total - s.processed);
                }
                else
                {
                    summary += "Cancelled: No\n";
                }

                summary += QString("\nResults folder:\n%1")
                    .arg(s.outputDirectory);
                summary += "\n\nCSV file:\nbatch_results.csv";

                QMessageBox::information(
                    this,
                    s.cancelled
                    ? "Batch Processing Cancelled"
                    : "Batch Processing Summary",
                    summary
                );
            }
        );
    }

    // --------------------------------------------------------
    // 3. NO POPUP: progress is shown in the existing labels,
    //    and the Start button becomes a Cancel button.
    // --------------------------------------------------------

    batchCancelRequested = false;
    batchAnimating = false;

    setProcessingState(true);

    if (btnStartBatch)
    {
        btnStartBatch->setText("Cancel Batch");
        btnStartBatch->setEnabled(true);
    }

    const QString methodName = getMethodName(selectedMethod);

    // --------------------------------------------------------
    // 4. THE WHOLE BATCH RUNS ON ONE WORKER THREAD
    // --------------------------------------------------------
    // RULE: never touch ui.* or batchProgress directly in here.

    batchWatcher->setFuture(
        QtConcurrent::run(
            [this,
            imagePaths,
            outputDirectory,
            correctedDirectory,
            previewDirectory,
            diagnosticDirectory,
            csvPath,
            selectedMethod,
            methodName]() -> BatchSummary
            {
                BatchSummary s;
                s.total = imagePaths.size();
                s.methodName = methodName;
                s.outputDirectory = outputDirectory;

                // ---- CSV ----
                QFile csvFile(csvPath);

                if (!csvFile.open(QIODevice::WriteOnly |
                    QIODevice::Truncate |
                    QIODevice::Text))
                {
                    s.errorMessage =
                        "Unable to create batch_results.csv.";
                    return s;
                }

                QTextStream csv(&csvFile);
                csv.setEncoding(QStringConverter::Utf8);

                csv << "filename,method,status,processing_time_ms,"
                    << "rotation_correction_deg,"
                    << "translation_x_px,translation_y_px,"
                    << "final_offset_x_px,final_offset_y_px,"
                    << "confidence,failure_reason\n";

                auto csvField = [](const QString& value) -> QString
                    {
                        QString escaped = value;
                        escaped.replace("\"", "\"\"");
                        return "\"" + escaped + "\"";
                    };

                const int total = imagePaths.size();

                // ---- PER-IMAGE LOOP ----
                for (int i = 0; i < total; ++i)
                {
                    if (batchCancelRequested.load())
                    {
                        s.cancelled = true;
                        break;
                    }

                    const QString imagePath = imagePaths[i];
                    const QFileInfo fileInfo(imagePath);
                    const QString originalFileName =
                        fileInfo.fileName();

                    const QString uniqueName =
                        QString("%1_%2")
                        .arg(i + 1, 4, 10, QChar('0'))
                        .arg(fileInfo.completeBaseName());

                    // Progress -> GUI thread.
                    QMetaObject::invokeMethod(
                        this,
                        [this, i, total, originalFileName]()
                        {
                            ui.lblImageStatus->setText(
                                QString("Batch Processing: Image %1 of %2\n%3")
                                .arg(i + 1)
                                .arg(total)
                                .arg(originalFileName)
                            );

                            ui.lblstatus->setText(
                                QString("Status : Processing %1 / %2 (%3%)")
                                .arg(i + 1)
                                .arg(total)
                                .arg(i * 100 / total)
                            );
                        },
                        Qt::QueuedConnection
                    );

                    QElapsedTimer timer;
                    timer.start();

                    bool imageSuccess = false;
                    QString failureReason;
                    CenteringResult result;
                    cv::Mat inputImage;

                    try
                    {
                        inputImage = cv::imread(
                            QFile::encodeName(imagePath).constData(),
                            cv::IMREAD_COLOR
                        );

                        const QString imageDiagnosticDirectory =
                            QDir(diagnosticDirectory).filePath(uniqueName);

                        if (inputImage.empty())
                        {
                            failureReason =
                                "Unable to read the input image.";
                        }
                        else if (!QDir().mkpath(imageDiagnosticDirectory))
                        {
                            failureReason =
                                "Unable to create diagnostic folder.";
                        }
                        else
                        {
                            result = processImageForCentering(
                                inputImage,
                                selectedMethod,
                                imageDiagnosticDirectory
                            );

                            if (!result.success)
                            {
                                failureReason =
                                    result.errorMessage.isEmpty()
                                    ? QString("Centering process failed.")
                                    : result.errorMessage;
                            }
                            else
                            {
                                const QString correctedPath =
                                    QDir(correctedDirectory).filePath(
                                        uniqueName + "_corrected.png");

                                const QString previewPath =
                                    QDir(previewDirectory).filePath(
                                        uniqueName + "_preview.png");

                                if (result.correctedImage.empty() ||
                                    !cv::imwrite(
                                        QFile::encodeName(correctedPath)
                                        .constData(),
                                        result.correctedImage))
                                {
                                    failureReason =
                                        "Unable to save corrected image.";
                                }
                                else if (result.previewImage.empty() ||
                                    !cv::imwrite(
                                        QFile::encodeName(previewPath)
                                        .constData(),
                                        result.previewImage))
                                {
                                    failureReason =
                                        "Unable to save preview image.";
                                }
                                else
                                {
                                    imageSuccess = true;
                                }
                            }
                        }
                    }
                    catch (const cv::Exception& e)
                    {
                        imageSuccess = false;
                        failureReason =
                            QString("OpenCV error: %1").arg(e.what());
                    }
                    catch (const std::exception& e)
                    {
                        imageSuccess = false;
                        failureReason =
                            QString("Processing error: %1").arg(e.what());
                    }
                    catch (...)
                    {
                        imageSuccess = false;
                        failureReason = "Unknown error occurred.";
                    }

                    const qint64 processingTime = timer.elapsed();

                    // Result -> GUI thread (small preview keeps it cheap).
                    if (imageSuccess)
                    {
                        // Downscale once so every animation frame is cheap.
                        cv::Mat small = result.previewImage;
                        cv::Mat originalSmall = inputImage;

                        const int longest =
                            std::max(small.cols, small.rows);

                        const double k =
                            longest > 1200 ? 1200.0 / longest : 1.0;

                        if (k < 1.0)
                        {
                            cv::resize(small, small, cv::Size(),
                                k, k, cv::INTER_AREA);

                            cv::resize(originalSmall, originalSmall,
                                cv::Size(), k, k, cv::INTER_AREA);
                        }

                        // ---- ORIGINAL -> ROTATED/CORRECTED ANIMATION ----
                        if (kBatchAnimationMs > 0 &&
                            !result.finalTransform.empty() &&
                            !originalSmall.empty() &&
                            !batchCancelRequested.load())
                        {
                            // Same rotation, translation scaled by k.
                            cv::Mat transformSmall =
                                result.finalTransform.clone();

                            transformSmall.at<double>(0, 2) *= k;
                            transformSmall.at<double>(1, 2) *= k;

                            const cv::RotatedRect boxSmall(
                                cv::Point2f(
                                    static_cast<float>(result.initialBox.center.x * k),
                                    static_cast<float>(result.initialBox.center.y * k)),
                                cv::Size2f(
                                    static_cast<float>(result.initialBox.size.width * k),
                                    static_cast<float>(result.initialBox.size.height * k)),
                                result.initialBox.angle
                            );

                            const cv::Point2f centerSmall(
                                static_cast<float>(result.initialCenter.x * k),
                                static_cast<float>(result.initialCenter.y * k)
                            );

                            batchAnimating = true;

                            QMetaObject::invokeMethod(
                                this,
                                [this, originalSmall, transformSmall,
                                boxSmall, centerSmall]()
                                {
                                    playBatchAnimation(
                                        originalSmall,
                                        transformSmall,
                                        boxSmall,
                                        centerSmall,
                                        kBatchAnimationMs
                                    );
                                },
                                Qt::QueuedConnection
                            );

                            // The worker waits here; the GUI thread is free.
                            // Bail out on cancel or if something goes wrong.
                            QElapsedTimer waitTimer;
                            waitTimer.start();

                            while (batchAnimating.load() &&
                                !batchCancelRequested.load() &&
                                waitTimer.elapsed() < kBatchAnimationMs + 3000)
                            {
                                QThread::msleep(15);
                            }
                        }

                        const QImage shown = bgrToQImage(small);
                        const double ox = result.finalOffsetX;
                        const double oy = result.finalOffsetY;
                        const double conf = result.confidence;

                        QMetaObject::invokeMethod(
                            this,
                            [this, shown, ox, oy, conf]()
                            {
                                displayImage(
                                    ui.originalImageLabel,
                                    shown
                                );

                                ui.lblstatus->setText(
                                    "Status : Centering Successful"
                                );

                                ui.labeloffsetx->setText(
                                    QString("Center Offset X : %1 px")
                                    .arg(ox, 0, 'f', 2)
                                );

                                ui.labeloffsety->setText(
                                    QString("Center Offset Y : %1 px")
                                    .arg(oy, 0, 'f', 2)
                                );

                                ui.labelconfidence->setText(
                                    conf >= 0.0
                                    ? QString("Confidence : %1 %")
                                    .arg(conf * 100.0, 0, 'f', 2)
                                    : QString("Confidence : N/A (OpenCV)")
                                );
                            },
                            Qt::QueuedConnection
                        );
                    }
                    else
                    {
                        QMetaObject::invokeMethod(
                            this,
                            [this, i, total, originalFileName, failureReason]()
                            {
                                ui.lblstatus->setText("Status : Failed");

                                ui.lblImageStatus->setText(
                                    QString("Image %1 of %2 failed:\n%3\n%4")
                                    .arg(i + 1)
                                    .arg(total)
                                    .arg(originalFileName)
                                    .arg(failureReason)
                                );
                            },
                            Qt::QueuedConnection
                        );
                    }

                    // Counters.
                    ++s.processed;

                    if (imageSuccess)
                        ++s.success;
                    else
                        ++s.failed;

                    // CSV row.
                    auto num = [&](double value) -> QString
                        {
                            return imageSuccess
                                ? QString::number(value, 'f', 4)
                                : QString();
                        };

                    QString confidenceValue = "N/A";

                    if (selectedMethod !=
                        DetectionMethod::GeometricRefinement &&
                        result.confidence >= 0.0)
                    {
                        confidenceValue =
                            QString::number(result.confidence, 'f', 4);
                    }

                    csv << csvField(originalFileName) << ","
                        << csvField(methodName) << ","
                        << csvField(imageSuccess ? "SUCCESS" : "FAILED") << ","
                        << processingTime << ","
                        << num(result.rotationCorrection) << ","
                        << num(result.translationX) << ","
                        << num(result.translationY) << ","
                        << num(result.finalOffsetX) << ","
                        << num(result.finalOffsetY) << ","
                        << csvField(confidenceValue) << ","
                        << csvField(failureReason)
                        << "\n";

                    csv.flush();
                }

                csv.flush();
                csvFile.close();

                return s;
            }
        )
    );
}


// ============================================================
// BATCH AUTO MODE - METHOD INFORMATION
// ============================================================

QString AutoCentering::getMethodName(DetectionMethod method) const
{
    switch (method)
    {
    case DetectionMethod::YOLO:
        return "YOLO";

    case DetectionMethod::GeometricRefinement:
        return "OpenCV Geometric Refinement";

    case DetectionMethod::YOLOGeometricRefinement:
        return "YOLO + Geometric Refinement";

    default:
        return "Unknown Method";
    }
}


QString AutoCentering::getMethodSuffix(DetectionMethod method) const
{
    switch (method)
    {
    case DetectionMethod::YOLO:
        return "YOLO";

    case DetectionMethod::GeometricRefinement:
        return "OpenCV";

    case DetectionMethod::YOLOGeometricRefinement:
        return "Hybrid";

    default:
        return "Unknown";
    }
}


// ============================================================
// BATCH AUTO MODE - PROCESS ONE IMAGE
// (runs on the worker thread: never touches ui.*)
// ============================================================

AutoCentering::CenteringResult
AutoCentering::processImageForCentering(
    const cv::Mat& inputImage,
    DetectionMethod selectedMethod,
    const QString& diagnosticDirectory
)
{
    CenteringResult output;

    try
    {
        // 1. CHECK INPUT IMAGE
        if (inputImage.empty())
        {
            output.errorMessage = "Input image is empty.";
            return output;
        }

        cv::Mat original = inputImage.clone();

        const int originalWidth = original.cols;
        const int originalHeight = original.rows;

        const cv::Point2f outputCenter(
            originalWidth / 2.0f,
            originalHeight / 2.0f
        );

        // 2. DIAGNOSTIC DIRECTORIES
        QString diagnosticRoot = diagnosticDirectory;

        if (diagnosticRoot.isEmpty())
        {
            diagnosticRoot =
                QDir::tempPath() +
                "/XRayAutoCenteringDiagnostics";
        }

        const QString initialDiagnostic =
            QDir(diagnosticRoot).filePath("initial");

        const QString rotatedDiagnostic =
            QDir(diagnosticRoot).filePath("rotated");

        const QString finalDiagnostic =
            QDir(diagnosticRoot).filePath("final");

        if (!QDir().mkpath(initialDiagnostic) ||
            !QDir().mkpath(rotatedDiagnostic) ||
            !QDir().mkpath(finalDiagnostic))
        {
            output.errorMessage =
                "Could not create diagnostic directories.";

            return output;
        }

        // 3. DETECTION RESULT STRUCTURE
        struct DetectionInfo
        {
            bool found = false;
            cv::Point2f center;
            cv::RotatedRect box;
            float confidence = -1.0f;
            QString message;
        };

        // 4. COMMON DETECTION FUNCTION
        auto detectObject =
            [this, selectedMethod](
                const cv::Mat& image,
                const QString& diagnosticPath
                ) -> DetectionInfo
            {
                DetectionInfo info;

                if (image.empty())
                {
                    info.message = "Detection input is empty.";
                    return info;
                }

                const std::string diagnostic =
                    diagnosticPath.toStdString();

                // METHOD 1: YOLO
                if (selectedMethod == DetectionMethod::YOLO)
                {
                    if (!detector || !detector->isLoaded())
                    {
                        info.message = "YOLO model is not loaded.";
                        return info;
                    }

                    const CableDetection detection =
                        detector->detect(image);

                    if (!detection.found)
                    {
                        info.message =
                            "YOLO could not detect the component.";
                        return info;
                    }

                    const float angleDegrees =
                        detection.angleRadians * 180.0f /
                        static_cast<float>(CV_PI);

                    info.box = cv::RotatedRect(
                        detection.center,
                        detection.size,
                        angleDegrees
                    );

                    info.center = detection.center;
                    info.confidence = detection.confidence;
                    info.found = true;
                    info.message = "YOLO detection successful.";
                    return info;
                }

                // METHOD 2: OPENCV GEOMETRIC REFINEMENT
                if (selectedMethod ==
                    DetectionMethod::GeometricRefinement)
                {
                    OpenCVOnlySegmentationResult result =
                        OpenCVOnlyGeometry::segmentFullImage(
                            image,
                            diagnostic
                        );

                    if (!result.success || !result.outerContourFound)
                    {
                        info.message =
                            QString::fromStdString(result.message);
                        return info;
                    }

                    const cv::RotatedRect box =
                        result.outerRotatedBoundingBox;

                    if (box.size.width <= 0.0f ||
                        box.size.height <= 0.0f)
                    {
                        info.message =
                            "OpenCV returned an invalid bounding box.";
                        return info;
                    }

                    info.box = box;
                    info.center = box.center;
                    info.confidence = -1.0f;
                    info.found = true;
                    info.message = "Geometric Refinement successful.";
                    return info;
                }

                // METHOD 3: YOLO + GEOMETRIC REFINEMENT
                if (selectedMethod ==
                    DetectionMethod::YOLOGeometricRefinement)
                {
                    if (!detector || !detector->isLoaded())
                    {
                        info.message = "YOLO model is not loaded.";
                        return info;
                    }

                    const CableDetection detection =
                        detector->detect(image);

                    if (!detection.found)
                    {
                        info.message =
                            "YOLO could not detect the component.";
                        return info;
                    }

                    const float angleDegrees =
                        detection.angleRadians * 180.0f /
                        static_cast<float>(CV_PI);

                    const cv::RotatedRect yoloBox(
                        detection.center,
                        detection.size,
                        angleDegrees
                    );

                    YOLOGuidedRefinementResult result =
                        OpenCVOnlyGeometry::refineWithinYOLORegion(
                            image,
                            yoloBox,
                            diagnostic,
                            0.25f
                        );

                    if (!result.success || !result.refinedContourFound)
                    {
                        info.message =
                            QString::fromStdString(result.message);
                        return info;
                    }

                    const cv::RotatedRect refinedBox =
                        result.refinedRotatedBoundingBox;

                    if (refinedBox.size.width <= 0.0f ||
                        refinedBox.size.height <= 0.0f)
                    {
                        info.message = "Invalid refined bounding box.";
                        return info;
                    }

                    info.box = refinedBox;
                    info.center = refinedBox.center;
                    info.confidence = detection.confidence;
                    info.found = true;
                    info.message =
                        "YOLO + Geometric Refinement successful.";
                    return info;
                }

                info.message = "No detection method selected.";
                return info;
            };

        // 5. INITIAL DETECTION
        const DetectionInfo initialDetection =
            detectObject(original, initialDiagnostic);

        if (!initialDetection.found)
        {
            output.errorMessage =
                "Initial detection failed: " +
                initialDetection.message;

            return output;
        }

        output.confidence = initialDetection.confidence;

        // 6. ROTATION CORRECTION
        const double correctionAngle =
            calculateRotationCorrection(initialDetection.box);

        output.rotationCorrection = correctionAngle;
        output.initialBox = initialDetection.box;
        output.initialCenter = initialDetection.center;

        // 7. ROTATE IMAGE WITHOUT CROPPING
        const cv::Point2f originalCenter(
            original.cols / 2.0f,
            original.rows / 2.0f
        );

        cv::Mat rotationMatrix =
            cv::getRotationMatrix2D(
                originalCenter,
                correctionAngle,
                1.0
            );

        const double cosine =
            std::abs(rotationMatrix.at<double>(0, 0));

        const double sine =
            std::abs(rotationMatrix.at<double>(0, 1));

        const int rotatedWidth = std::max(
            1,
            static_cast<int>(
                std::ceil(original.rows * sine + original.cols * cosine))
        );

        const int rotatedHeight = std::max(
            1,
            static_cast<int>(
                std::ceil(original.rows * cosine + original.cols * sine))
        );

        rotationMatrix.at<double>(0, 2) +=
            rotatedWidth / 2.0 - originalCenter.x;

        rotationMatrix.at<double>(1, 2) +=
            rotatedHeight / 2.0 - originalCenter.y;

        // 8. DETECT AFTER ROTATION
        DetectionInfo rotatedDetection;

        if (correctionAngle == 0.0)
        {
            // No rotation: reuse the first detection.
            rotatedDetection = initialDetection;
        }
        else
        {
            cv::Mat rotated;

            cv::warpAffine(
                original,
                rotated,
                rotationMatrix,
                cv::Size(rotatedWidth, rotatedHeight),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(255, 255, 255)
            );

            rotatedDetection =
                detectObject(rotated, rotatedDiagnostic);
        }

        if (!rotatedDetection.found)
        {
            output.errorMessage =
                "Detection after rotation failed: " +
                rotatedDetection.message;

            return output;
        }

        // 9. TRANSLATION
        const double shiftX =
            outputCenter.x - rotatedDetection.center.x;

        const double shiftY =
            outputCenter.y - rotatedDetection.center.y;

        output.translationX = shiftX;
        output.translationY = shiftY;

        // 10. COMBINE ROTATION AND TRANSLATION
        cv::Mat finalTransform = rotationMatrix.clone();

        finalTransform.at<double>(0, 2) += shiftX;
        finalTransform.at<double>(1, 2) += shiftY;

        // 11. CORRECTED IMAGE
        cv::Mat corrected;

        cv::warpAffine(
            original,
            corrected,
            finalTransform,
            cv::Size(originalWidth, originalHeight),
            cv::INTER_CUBIC,
            cv::BORDER_CONSTANT,
            cv::Scalar(255, 255, 255)
        );

        // 12. FINAL DETECTION
        const DetectionInfo finalDetection =
            detectObject(corrected, finalDiagnostic);

        if (!finalDetection.found)
        {
            output.errorMessage =
                "Final detection failed: " +
                finalDetection.message;

            return output;
        }

        // 13. FINAL RESIDUAL OFFSET
        output.finalOffsetX =
            finalDetection.center.x - outputCenter.x;

        output.finalOffsetY =
            finalDetection.center.y - outputCenter.y;

        // 14. FINAL PREVIEW
        cv::Mat preview = corrected.clone();

        drawRotatedBox(
            preview,
            finalDetection.box,
            cv::Scalar(0, 255, 0),
            3
        );

        cv::circle(
            preview,
            finalDetection.center,
            6,
            cv::Scalar(0, 0, 255),
            cv::FILLED,
            cv::LINE_AA
        );

        cv::drawMarker(
            preview,
            cv::Point(originalWidth / 2, originalHeight / 2),
            cv::Scalar(255, 0, 0),
            cv::MARKER_CROSS,
            25,
            2,
            cv::LINE_AA
        );

        // 15. RETURN RESULTS
        output.finalTransform = finalTransform.clone();
        output.correctedImage = corrected;
        output.previewImage = preview;
        output.success = true;
        output.errorMessage = "";

        return output;
    }
    catch (const cv::Exception& exception)
    {
        output.success = false;

        output.errorMessage =
            QString("OpenCV error: %1")
            .arg(QString::fromLocal8Bit(exception.what()));

        return output;
    }
    catch (const std::exception& exception)
    {
        output.success = false;

        output.errorMessage =
            QString("Processing error: %1")
            .arg(QString::fromLocal8Bit(exception.what()));

        return output;
    }
    catch (...)
    {
        output.success = false;

        output.errorMessage =
            "Unknown error occurred during image processing.";

        return output;
    }
}


// ============================================================
// PROCESSING STATE
// ============================================================

void AutoCentering::setProcessingState(bool processing)
{
    isProcessing = processing;

    // Disable controls while background processing is running.
    ui.btnLoadImage->setEnabled(!processing);
    ui.btnProcess->setEnabled(!processing);
    ui.btn_OpenCVOnly->setEnabled(!processing);
    ui.btn_yologf->setEnabled(!processing);
    ui.btnAutoCenter->setEnabled(!processing);

    if (btnBrowseBatchFolder)
        btnBrowseBatchFolder->setEnabled(!processing);

    if (btnStartBatch)
        btnStartBatch->setEnabled(!processing);

    if (cmbBatchMethod)
        cmbBatchMethod->setEnabled(!processing);

    if (processing)
    {
        ui.lblImageStatus->setText(
            "Processing in background. Please wait..."
        );
    }
}


// ============================================================
// BATCH AUTO MODE - PER-IMAGE ANIMATION (GUI THREAD)
// ============================================================
//
// Timer-driven, no nested event loop. Animates the original image
// rotating and shifting into the corrected position, with the
// detected box and centre moving with it.
// ============================================================

void AutoCentering::playBatchAnimation(
    const cv::Mat& originalSmall,
    const cv::Mat& transformSmall,
    const cv::RotatedRect& boxSmall,
    const cv::Point2f& centerSmall,
    int durationMs)
{
    if (originalSmall.empty() || transformSmall.empty())
    {
        batchAnimating = false;
        return;
    }

    const int frameWidth = originalSmall.cols;
    const int frameHeight = originalSmall.rows;

    const cv::Point2f targetCenter(
        frameWidth / 2.0f,
        frameHeight / 2.0f
    );

    auto* animation = new QVariantAnimation(this);

    animation->setDuration(durationMs);
    animation->setStartValue(0.0);
    animation->setEndValue(1.0);
    animation->setEasingCurve(QEasingCurve::InOutSine);

    connect(
        animation,
        &QVariantAnimation::valueChanged,
        this,
        [this,
        originalSmall,
        transformSmall,
        boxSmall,
        centerSmall,
        frameWidth,
        frameHeight,
        targetCenter](const QVariant& value)
        {
            const double progress = value.toDouble();

            // Interpolate identity -> final affine transform.
            cv::Mat currentTransform = cv::Mat::eye(2, 3, CV_64F);

            for (int row = 0; row < 2; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    const double startValue =
                        row == column ? 1.0 : 0.0;

                    const double finalValue =
                        transformSmall.at<double>(row, column);

                    currentTransform.at<double>(row, column) =
                        startValue +
                        (finalValue - startValue) * progress;
                }
            }

            cv::Mat frame;

            cv::warpAffine(
                originalSmall,
                frame,
                currentTransform,
                cv::Size(frameWidth, frameHeight),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(255, 255, 255)
            );

            auto apply = [&currentTransform](const cv::Point2f& p)
                {
                    return cv::Point2f(
                        static_cast<float>(
                            currentTransform.at<double>(0, 0) * p.x +
                            currentTransform.at<double>(0, 1) * p.y +
                            currentTransform.at<double>(0, 2)),
                        static_cast<float>(
                            currentTransform.at<double>(1, 0) * p.x +
                            currentTransform.at<double>(1, 1) * p.y +
                            currentTransform.at<double>(1, 2))
                    );
                };

            // Moving detection box.
            cv::Point2f corners[4];

            boxSmall.points(corners);

            for (int i = 0; i < 4; ++i)
                corners[i] = apply(corners[i]);

            for (int i = 0; i < 4; ++i)
            {
                cv::line(
                    frame,
                    corners[i],
                    corners[(i + 1) % 4],
                    cv::Scalar(0, 255, 0),
                    2,
                    cv::LINE_AA
                );
            }

            // Moving component centre.
            const cv::Point2f movingCenter = apply(centerSmall);

            cv::circle(
                frame,
                movingCenter,
                5,
                cv::Scalar(0, 0, 255),
                cv::FILLED,
                cv::LINE_AA
            );

            // Fixed target centre.
            cv::drawMarker(
                frame,
                cv::Point(
                    static_cast<int>(targetCenter.x),
                    static_cast<int>(targetCenter.y)),
                cv::Scalar(255, 0, 0),
                cv::MARKER_CROSS,
                20,
                2,
                cv::LINE_AA
            );

            displayImage(
                ui.originalImageLabel,
                bgrToQImage(frame)
            );

            ui.labeloffsetx->setText(
                QString("Center Offset X : %1 px")
                .arg(movingCenter.x - targetCenter.x, 0, 'f', 1)
            );

            ui.labeloffsety->setText(
                QString("Center Offset Y : %1 px")
                .arg(movingCenter.y - targetCenter.y, 0, 'f', 1)
            );

            ui.lblstatus->setText(
                QString("Status : Auto Centering %1%")
                .arg(static_cast<int>(progress * 100.0))
            );
        }
    );

    connect(
        animation,
        &QVariantAnimation::finished,
        this,
        [this, animation]()
        {
            batchAnimating = false;
            animation->deleteLater();
        }
    );

    animation->start();
}


// ============================================================
// SAVE RESULT (SINGLE-IMAGE AUTO CENTER)
// ============================================================
//
// Auto Center keeps its result in memory. Files are only written
// here, when the user clicks "Save Result".
// ============================================================

void AutoCentering::saveResult()
{
    if (pendingCorrectedImage.empty())
    {
        QMessageBox::information(
            this,
            "Save Result",
            "There is no centered image to save yet.\n"
            "Load an image and run Auto Center first."
        );

        return;
    }

    const QString defaultPath =
        QDir(QDir::currentPath() + "/output/auto_centering")
        .filePath(pendingBaseName + "_corrected.png");

    QString correctedPath = QFileDialog::getSaveFileName(
        this,
        "Save Centered Image",
        defaultPath,
        "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;"
        "BMP Image (*.bmp);;TIFF Image (*.tif *.tiff)"
    );

    if (correctedPath.isEmpty())
        return;

    if (QFileInfo(correctedPath).suffix().isEmpty())
        correctedPath += ".png";

    const QFileInfo correctedInfo(correctedPath);

    if (!QDir().mkpath(correctedInfo.absolutePath()))
    {
        QMessageBox::warning(
            this,
            "Save Error",
            "Could not create the output folder."
        );

        return;
    }

    // The preview (with the detection box) is saved next to it.
    QString previewStem = correctedInfo.completeBaseName();

    if (previewStem.endsWith("_corrected"))
        previewStem.chop(QString("_corrected").size());

    const QString previewPath =
        correctedInfo.absoluteDir().filePath(
            previewStem + "_preview.png"
        );

    try
    {
        if (!cv::imwrite(
            QFile::encodeName(correctedPath).constData(),
            pendingCorrectedImage))
        {
            QMessageBox::warning(
                this,
                "Save Error",
                "Could not save the corrected image."
            );

            return;
        }

        if (!pendingPreviewImage.empty() &&
            !cv::imwrite(
                QFile::encodeName(previewPath).constData(),
                pendingPreviewImage))
        {
            QMessageBox::warning(
                this,
                "Save Error",
                "The corrected image was saved, but the preview "
                "image could not be saved."
            );

            return;
        }
    }
    catch (const cv::Exception& exception)
    {
        QMessageBox::critical(
            this,
            "Save Error",
            QString::fromLocal8Bit(exception.what())
        );

        return;
    }

    ui.lblstatus->setText("Status : Result Saved");

    ui.lblImageStatus->setText(
        QString(
            "Saved.\n\n"
            "Corrected image:\n%1\n\n"
            "Preview image:\n%2"
        )
        .arg(QDir::toNativeSeparators(correctedPath))
        .arg(QDir::toNativeSeparators(previewPath))
    );
}