#include "MainWindow.h"
#include "OpenCVOnlyGeometry.h"

#include <QFile>
#include <QApplication>
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

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <exception>


#include <QCoreApplication>

#include <QStringConverter>

#include <QThread>
#include <QTransform>

// ============================================================
// HELPER FUNCTION FORWARD DECLARATIONS
// ============================================================

namespace
{
    QImage bgrToQImage(const cv::Mat& image);
}

void AutoCentering::processBatchImages(
    const QStringList& imagePaths,
    const QString& outputDirectory,
    DetectionMethod selectedMethod)
{
    // ========================================================
    // 1. CREATE OUTPUT FOLDERS
    // ========================================================

    QDir outputDir(outputDirectory);

    const QString correctedDirectory =
        outputDir.filePath("corrected");

    const QString previewDirectory =
        outputDir.filePath("previews");

    const QString diagnosticDirectory =
        outputDir.filePath("diagnostics");

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

    // ========================================================
    // 2. CREATE CSV FILE
    // ========================================================

    const QString csvPath =
        outputDir.filePath("batch_results.csv");

    QFile csvFile(csvPath);

    if (!csvFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        QMessageBox::critical(
            this,
            "CSV Error",
            "Unable to create batch_results.csv."
        );

        return;
    }

    QTextStream csv(&csvFile);
    csv.setEncoding(QStringConverter::Utf8);

    csv << "filename,method,status,processing_time_ms,"
        << "rotation_correction_deg,"
        << "translation_x_px,translation_y_px,"
        << "final_offset_x_px,final_offset_y_px,"
        << "confidence,failure_reason\n";

    // CSV escaping helper
    auto csvField = [](const QString& value) -> QString
        {
            QString escaped = value;
            escaped.replace("\"", "\"\"");
            return "\"" + escaped + "\"";
        };

    // ========================================================
    // 3. CREATE PROGRESS DIALOG
    // ========================================================

    QProgressDialog progress(
        "Preparing batch processing...",
        "Cancel",
        0,
        imagePaths.size(),
        this
    );

    progress.setWindowTitle("Batch Auto-Centering");
    progress.setWindowModality(Qt::WindowModal);
    progress.setMinimumDuration(0);
    progress.setValue(0);

    // ========================================================
    // 4. PROCESSING COUNTERS
    // ========================================================

    int successCount = 0;
    int failedCount = 0;
    int processedCount = 0;

    bool cancelled = false;

    const QString methodName =
        getMethodName(selectedMethod);

    // ========================================================
    // 5. PROCESS EACH IMAGE
    // ========================================================

    for (int i = 0; i < imagePaths.size(); ++i)
    {
        QCoreApplication::processEvents();

        if (progress.wasCanceled())
        {
            cancelled = true;
            break;
        }

        const QString imagePath = imagePaths[i];

        const QFileInfo fileInfo(imagePath);

        const QString originalFileName =
            fileInfo.fileName();

        // Add an index to avoid duplicate filenames
        // when different files have the same basename.

        const QString uniqueName =
            QString("%1_%2")
            .arg(i + 1, 4, 10, QChar('0'))
            .arg(fileInfo.completeBaseName());

        progress.setLabelText(
            QString("Processing image %1 of %2:\n%3")
            .arg(i + 1)
            .arg(imagePaths.size())
            .arg(originalFileName)
        );

        progress.setValue(i);

        QCoreApplication::processEvents();

        QElapsedTimer timer;
        timer.start();

        bool imageSuccess = false;

        CenteringResult result;

        QString failureReason;

        // ====================================================
        // 5A. READ ORIGINAL IMAGE
        // ====================================================

        cv::Mat inputImage;

        try
        {
            const QByteArray inputPathBytes =
                QFile::encodeName(imagePath);
            inputImage = cv::imread(
                inputPathBytes.constData(),
                cv::IMREAD_COLOR
            );

            if (inputImage.empty())
            {
                failureReason =
                    "Unable to read the input image.";
            }
            else
            {
                // ============================================
                // DISPLAY CURRENT ORIGINAL IMAGE
                // ============================================

                QImage originalDisplay = bgrToQImage(inputImage).copy();



                ui.originalImageLabel->setPixmap(
                    QPixmap::fromImage(originalDisplay).scaled(
                        ui.originalImageLabel->size(),
                        Qt::KeepAspectRatio,
                        Qt::SmoothTransformation
                    )
                );

                ui.lblImageStatus->setText(
                    QString("Batch Processing: Image %1 of %2\n%3")
                    .arg(i + 1)
                    .arg(imagePaths.size())
                    .arg(originalFileName)
                );

                ui.lblstatus->setText("Status: Processing...");
                ui.labeloffsetx->setText("Center Offset X: Processing...");
                ui.labeloffsety->setText("Center Offset Y: Processing...");
                ui.labelconfidence->setText("Confidence: Processing...");

                QCoreApplication::processEvents();

                // ============================================
                // 5B. CREATE IMAGE DIAGNOSTIC FOLDER
                // ============================================

                const QString imageDiagnosticDirectory =
                    QDir(diagnosticDirectory).filePath(uniqueName);

                if (!QDir().mkpath(imageDiagnosticDirectory))
                {
                    failureReason =
                        "Unable to create diagnostic folder.";
                }
                else
                {
                    // ========================================
                    // 5C. RUN SELECTED CENTERING METHOD
                    // ========================================

                    result = processImageForCentering(
                        inputImage,
                        selectedMethod,
                        imageDiagnosticDirectory
                    );

                    if (!result.success)
                    {
                        failureReason = result.errorMessage;

                        if (failureReason.isEmpty())
                        {
                            failureReason =
                                "Centering process failed.";
                        }
                    }
                    else
                    {
                        // ====================================
                        // 5D. SAVE CORRECTED IMAGE
                        // ====================================

                        const QString correctedPath =
                            QDir(correctedDirectory).filePath(
                                uniqueName + "_corrected.png"
                            );

                        const QByteArray correctedPathBytes =
                            QFile::encodeName(correctedPath);

                        const bool correctedSaved =
                            !result.correctedImage.empty() &&
                            cv::imwrite(
                                correctedPathBytes.constData(),
                                result.correctedImage
                            );

                        // ====================================
                        // 5E. SAVE PREVIEW IMAGE
                        // ====================================

                        const QString previewPath =
                            QDir(previewDirectory).filePath(
                                uniqueName + "_preview.png"
                            );

                        const QByteArray previewPathBytes =
                            QFile::encodeName(previewPath);

                        const bool previewSaved =
                            !result.previewImage.empty() &&
                            cv::imwrite(
                                previewPathBytes.constData(),
                                result.previewImage
                            );

                        if (!correctedSaved)
                        {
                            failureReason =
                                "Unable to save corrected image.";
                        }
                        else if (!previewSaved)
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
        }
        catch (const cv::Exception& e)
        {
            failureReason =
                QString("OpenCV error: %1").arg(e.what());
        }
        catch (const std::exception& e)
        {
            failureReason =
                QString("Processing error: %1").arg(e.what());
        }
        catch (...)
        {
            failureReason =
                "Unknown error occurred.";
        }

        const qint64 processingTime =
            timer.elapsed();


        // ====================================================
        // 5F. GRADUAL ROTATION ANIMATION
        // ====================================================

        if (imageSuccess && !inputImage.empty())
        {
            const int animationFrames = 60;
            const int frameDelayMs = 25;

            const double rotationAngle =
                result.rotationCorrection;

            // Show the original image before animation.

            QPixmap originalPixmap =
                QPixmap::fromImage(
                    bgrToQImage(inputImage).copy()
                );

            for (int frame = 1;
                frame <= animationFrames;
                ++frame)
            {
                const double progress =
                    static_cast<double>(frame) /
                    animationFrames;

                // OpenCV and Qt use different visual
                // rotation directions for positive angles.

                const double currentAngle =
                    -rotationAngle * progress;

                QTransform transform;

                transform.rotate(currentAngle);

                QPixmap rotatedPixmap =
                    originalPixmap.transformed(
                        transform,
                        Qt::SmoothTransformation
                    );

                ui.originalImageLabel->setPixmap(
                    rotatedPixmap.scaled(
                        ui.originalImageLabel->size(),
                        Qt::KeepAspectRatio,
                        Qt::SmoothTransformation
                    )
                );

                ui.lblImageStatus->setText(
                    QString(
                        "Batch Processing: Image %1 of %2\n"
                        "Rotating: %3 degrees"
                    )
                    .arg(i + 1)
                    .arg(imagePaths.size())
                    .arg(currentAngle, 0, 'f', 2)
                );

                QCoreApplication::processEvents();

                QThread::msleep(frameDelayMs);
            }

            // ================================================
            // DISPLAY FINAL CORRECTED IMAGE
            // ================================================

            QImage correctedDisplay =
                bgrToQImage(result.previewImage).copy();

            ui.originalImageLabel->setPixmap(
                QPixmap::fromImage(correctedDisplay).scaled(
                    ui.originalImageLabel->size(),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation
                )
            );

            ui.lblstatus->setText(
                "Status: Centering Successful"
            );

            ui.labeloffsetx->setText(
                QString("Center Offset X: %1 px")
                .arg(result.finalOffsetX, 0, 'f', 2)
            );

            ui.labeloffsety->setText(
                QString("Center Offset Y: %1 px")
                .arg(result.finalOffsetY, 0, 'f', 2)
            );

            if (result.confidence >= 0.0)
            {
                ui.labelconfidence->setText(
                    QString("Confidence: %1%")
                    .arg(result.confidence * 100.0, 0, 'f', 2)
                );
            }
            else
            {
                ui.labelconfidence->setText(
                    "Confidence: N/A"
                );
            }

            QCoreApplication::processEvents();
        }


        if (!imageSuccess)
        {
            ui.lblstatus->setText(
                "Status: Centering Failed"
            );

            ui.lblImageStatus->setText(
                QString("Failed: %1\nReason: %2")
                .arg(originalFileName)
                .arg(failureReason)
            );

            QCoreApplication::processEvents();
        }
        // ====================================================
        // 6. UPDATE COUNTERS
        // ====================================================

        processedCount++;

        if (imageSuccess)
        {
            successCount++;
        }
        else
        {
            failedCount++;
        }

        // ====================================================
        // 7. WRITE CSV RESULT
        // ====================================================

        const QString status =
            imageSuccess ? "SUCCESS" : "FAILED";

        const QString rotationValue =
            imageSuccess
            ? QString::number(result.rotationCorrection, 'f', 4)
            : "";

        const QString translationXValue =
            imageSuccess
            ? QString::number(result.translationX, 'f', 4)
            : "";

        const QString translationYValue =
            imageSuccess
            ? QString::number(result.translationY, 'f', 4)
            : "";

        const QString finalOffsetXValue =
            imageSuccess
            ? QString::number(result.finalOffsetX, 'f', 4)
            : "";

        const QString finalOffsetYValue =
            imageSuccess
            ? QString::number(result.finalOffsetY, 'f', 4)
            : "";

        QString confidenceValue = "N/A";

        if (selectedMethod != DetectionMethod::GeometricRefinement)
        {
            confidenceValue =
                result.confidence >= 0.0
                ? QString::number(result.confidence, 'f', 4)
                : "N/A";
        }

        csv
            << csvField(originalFileName) << ","
            << csvField(methodName) << ","
            << csvField(status) << ","
            << processingTime << ","
            << rotationValue << ","
            << translationXValue << ","
            << translationYValue << ","
            << finalOffsetXValue << ","
            << finalOffsetYValue << ","
            << csvField(confidenceValue) << ","
            << csvField(failureReason)
            << "\n";

        csv.flush();

        // ====================================================
        // 8. UPDATE PROGRESS
        // ====================================================

        progress.setValue(i + 1);

        QCoreApplication::processEvents();

        if (progress.wasCanceled())
        {
            cancelled = true;
            break;
        }
    }

    // ========================================================
    // 9. CLOSE CSV FILE
    // ========================================================

    csv.flush();
    csvFile.close();

    // ========================================================
    // 10. FINAL SUMMARY
    // ========================================================

    progress.setValue(imagePaths.size());

    const int remainingCount =
        imagePaths.size() - processedCount;

    QString summary;

    summary += "Batch processing completed.\n\n";

    summary += QString("Selected method: %1\n")
        .arg(methodName);

    summary += QString("Total images: %1\n")
        .arg(imagePaths.size());

    summary += QString("Processed: %1\n")
        .arg(processedCount);

    summary += QString("Successful: %1\n")
        .arg(successCount);

    summary += QString("Failed: %1\n")
        .arg(failedCount);

    if (cancelled)
    {
        summary += QString("Cancelled: Yes\n");
        summary += QString("Not processed: %1\n")
            .arg(remainingCount);
    }
    else
    {
        summary += "Cancelled: No\n";
    }

    summary += QString("\nResults folder:\n%1")
        .arg(outputDirectory);

    summary += "\n\nCSV file:\nbatch_results.csv";

    QMessageBox::information(
        this,
        cancelled
        ? "Batch Processing Cancelled"
        : "Batch Processing Summary",
        summary
    );
}
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

    cmbBatchMethod->addItem("YOLO", static_cast<int>(DetectionMethod::YOLO));
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
// HELPER FUNCTIONS
// ============================================================

namespace
{
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

    // ========================================================
    // IMAGE PREVIEW LAYOUT
    // ========================================================

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

    ui.originalImageLabel->setAlignment(
        Qt::AlignCenter
    );
    ui.lblimagetitle->setSizePolicy(
        QSizePolicy::Preferred,
        QSizePolicy::Fixed
    );

    ui.lblimagetitle->setFixedHeight(32);

    ui.lblimagetitle->setAlignment(
        Qt::AlignLeft | Qt::AlignVCenter
    );

    // Load Dark Minimal UI Theme
    QFile styleFile(":/styles/dark_minimal.qss");

    if (styleFile.open(QFile::ReadOnly | QFile::Text))
    {
        QString styleSheet = QString::fromUtf8(styleFile.readAll());

        qApp->setStyleSheet(styleSheet);

        styleFile.close();

        qDebug() << "Dark Minimal Theme Loaded Successfully";
    }
    else
    {
        qWarning() << "Failed to load Dark Minimal Theme";
    }

    detector = new YOLODetector(L"models/best.onnx");

    connect(
        ui.btnLoadImage,
        &QPushButton::clicked,
        this,
        &AutoCentering::loadImage
    );

    connect(
        ui.btnProcess,
        &QPushButton::clicked,
        this,
        &AutoCentering::processImage
    );

    connect(
        ui.btn_OpenCVOnly,
        &QPushButton::clicked,
        this,
        &AutoCentering::processOpenCVOnly
    );

    connect(
        ui.btnAutoCenter,
        &QPushButton::clicked,
        this,
        &AutoCentering::autoCenterImage
    );

    connect(
        ui.btn_yologf,
        &QPushButton::clicked,
        this,
        &AutoCentering::processYOLOGeometricRefinement
        
    );
    //configure batch auto mode 
    setupBatchAutoMode();

    
}


// ============================================================
// DESTRUCTOR
// ============================================================

AutoCentering::~AutoCentering()
{
    if (shiftAnimation)
    {
        shiftAnimation->stop();
        shiftAnimation = nullptr;
    }

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

    // Reset detection method.
    activeMethod = DetectionMethod::None;

    displayImage(
        ui.originalImageLabel,
        currentImage
    );

    ui.lblImageStatus->setText(
        "Image Loaded Successfully"
    );

    ui.lblstatus->setText(
        "Status : Waiting"
    );

    ui.labeloffsetx->setText(
        "Center Offset X : 0 px"
    );

    ui.labeloffsety->setText(
        "Center Offset Y : 0 px"
    );

    ui.labelconfidence->setText(
        "Confidence : --"
    );
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
            ui.lblstatus->setText(
                "Status : No Detection"
            );

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

        const double imageCenterX =
            bgr.cols / 2.0;

        const double imageCenterY =
            bgr.rows / 2.0;

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
            .arg(
                detection.confidence * 100.0f,
                0,
                'f',
                2
            )
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
// OPENCV-ONLY PROCESSING
// ============================================================

void AutoCentering::processOpenCVOnly()
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

    activeMethod = DetectionMethod::GeometricRefinement;

    ui.lblstatus->setText(
        "Status : Geometric Refinement Selected"
    );

    ui.lblImageStatus->setText(
        "Detecting the outer boundary of the main component..."
    );

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

        OpenCVOnlySegmentationResult result =
            OpenCVOnlyGeometry::segmentFullImage(
                bgr,
                outputFolder.toStdString()
            );

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

        const QString outputPath =
            QDir(outputFolder).filePath(
                "opencv_main_component_result.png"
            );

        if (!cv::imwrite(
            outputPath.toStdString(),
            result.candidateVisualization))
        {
            QMessageBox::warning(
                this,
                "Save Error",
                "Could not save the OpenCV result."
            );

            return;
        }

        displayImage(
            ui.originalImageLabel,
            bgrToQImage(result.candidateVisualization)
        );

        const double imageCenterX =
            bgr.cols / 2.0;

        const double imageCenterY =
            bgr.rows / 2.0;

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
                "Saved: %9"
            )
            .arg(result.outerCableContour.size())
            .arg(box.size.width, 0, 'f', 1)
            .arg(box.size.height, 0, 'f', 1)
            .arg(box.angle, 0, 'f', 2)
            .arg(box.center.x, 0, 'f', 1)
            .arg(box.center.y, 0, 'f', 1)
            .arg(offsetX, 0, 'f', 1)
            .arg(offsetY, 0, 'f', 1)
            .arg(outputPath)
        );
    }
    catch (const cv::Exception& exception)
    {
        ui.lblstatus->setText("Status : OpenCV Error");

        QMessageBox::critical(
            this,
            "OpenCV Error",
            QString::fromLocal8Bit(exception.what())
        );
    }
    catch (const std::exception& exception)
    {
        ui.lblstatus->setText("Status : Processing Error");

        QMessageBox::critical(
            this,
            "Processing Error",
            QString::fromLocal8Bit(exception.what())
        );
    }
    catch (...)
    {
        ui.lblstatus->setText("Status : Failed");

        QMessageBox::critical(
            this,
            "Processing Error",
            "An unknown error occurred."
        );
    }
}


// ============================================================
// AUTO CENTERING - YOLO + GEOMETRIC REFINEMENT
// ============================================================
// ============================================================
// YOLO + GEOMETRIC REFINEMENT DETECTION
// ============================================================

void AutoCentering::processYOLOGeometricRefinement()
{
    activeMethod = DetectionMethod::YOLOGeometricRefinement;
    // --------------------------------------------------------
    // 1. CHECK IMAGE
    // --------------------------------------------------------

    if (currentImage.isNull())
    {
        QMessageBox::warning(
            this,
            "No Image",
            "Please load an X-ray image first."
        );

        return;
    }

    // --------------------------------------------------------
    // 2. CHECK YOLO MODEL
    // --------------------------------------------------------

    if (!detector || !detector->isLoaded())
    {
        ui.lblstatus->setText(
            "Status : YOLO Model Error"
        );

        QMessageBox::critical(
            this,
            "YOLO Error",
            "The ONNX model is not loaded."
        );

        return;
    }

    try
    {
        // ----------------------------------------------------
        // 3. CONVERT QIMAGE TO OPENCV
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // 4. YOLO DETECTION
        // ----------------------------------------------------

        ui.lblstatus->setText(
            "Status : YOLO Detecting Component..."
        );

        const CableDetection detection =
            detector->detect(bgr);

        if (!detection.found)
        {
            ui.lblstatus->setText(
                "Status : YOLO Detection Failed"
            );

            ui.lblImageStatus->setText(
                "YOLO could not detect the component."
            );

            QMessageBox::information(
                this,
                "YOLO Detection",
                "No component passed the YOLO confidence threshold."
            );

            return;
        }

        // Convert YOLO angle from radians to degrees.
        const float angleDegrees =
            detection.angleRadians * 180.0f /
            static_cast<float>(CV_PI);

        // Create YOLO rotated bounding box.
        const cv::RotatedRect yoloBox(
            detection.center,
            detection.size,
            angleDegrees
        );

        // ----------------------------------------------------
        // 5. CREATE OUTPUT DIRECTORY
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // 6. YOLO-GUIDED OPENCV REFINEMENT
        // ----------------------------------------------------

        ui.lblstatus->setText(
            "Status : Refining Component with OpenCV..."
        );

        YOLOGuidedRefinementResult result =
            OpenCVOnlyGeometry::refineWithinYOLORegion(
                bgr,
                yoloBox,
                outputFolder.toStdString(),
                0.25f
            );

        if (!result.success ||
            !result.refinedContourFound)
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

        // ----------------------------------------------------
        // 7. GET REFINED GEOMETRY
        // ----------------------------------------------------

        const cv::RotatedRect refinedBox =
            result.refinedRotatedBoundingBox;

        if (refinedBox.size.width <= 0.0f ||
            refinedBox.size.height <= 0.0f)
        {
            QMessageBox::warning(
                this,
                "Invalid Bounding Box",
                "OpenCV returned an invalid refined bounding box."
            );

            return;
        }

        // ----------------------------------------------------
        // 8. CHECK VISUALIZATION
        // ----------------------------------------------------

        if (result.fullImageVisualization.empty())
        {
            QMessageBox::warning(
                this,
                "Display Error",
                "The hybrid result visualization is empty."
            );

            return;
        }

        // ----------------------------------------------------
        // 9. SAVE RESULT IMAGE
        // ----------------------------------------------------

        const QString outputPath =
            QDir(outputFolder).filePath(
                "yolo_geometric_refinement_result.png"
            );

        if (!cv::imwrite(
            outputPath.toStdString(),
            result.fullImageVisualization))
        {
            QMessageBox::warning(
                this,
                "Save Error",
                "Could not save the hybrid result image."
            );

            return;
        }

        // ----------------------------------------------------
        // 10. DISPLAY RESULT
        // ----------------------------------------------------

        displayImage(
            ui.originalImageLabel,
            bgrToQImage(result.fullImageVisualization)
        );

        // ----------------------------------------------------
        // 11. CALCULATE COMPONENT OFFSET
        // ----------------------------------------------------

        const double imageCenterX =
            bgr.cols / 2.0;

        const double imageCenterY =
            bgr.rows / 2.0;

        const double offsetX =
            refinedBox.center.x - imageCenterX;

        const double offsetY =
            refinedBox.center.y - imageCenterY;

        // ----------------------------------------------------
        // 12. UPDATE OFFSET LABELS
        // ----------------------------------------------------

        ui.labeloffsetx->setText(
            QString("Center Offset X : %1 px")
            .arg(offsetX, 0, 'f', 1)
        );

        ui.labeloffsety->setText(
            QString("Center Offset Y : %1 px")
            .arg(offsetY, 0, 'f', 1)
        );

        // ----------------------------------------------------
        // 13. UPDATE CONFIDENCE
        // ----------------------------------------------------

        ui.labelconfidence->setText(
            QString("YOLO Confidence : %1 %")
            .arg(
                detection.confidence * 100.0f,
                0,
                'f',
                2
            )
        );

        // ----------------------------------------------------
        // 14. UPDATE STATUS
        // ----------------------------------------------------

        ui.lblstatus->setText(
            "Status : YOLO + Geometric Refinement Completed"
        );

        ui.lblImageStatus->setText(
            QString(
                "YOLO + Geometric Refinement Successful.\n\n"

                "YOLO Confidence: %1 %\n\n"

                "YOLO ROI:\n"
                "X: %2 px\n"
                "Y: %3 px\n"
                "Width: %4 px\n"
                "Height: %5 px\n\n"

                "Refined Component:\n"
                "Contour Points: %6\n"
                "Width: %7 px\n"
                "Height: %8 px\n"
                "Angle: %9 degrees\n\n"

                "Refined Centre:\n"
                "X: %10 px\n"
                "Y: %11 px\n\n"

                "Image Centre:\n"
                "X: %12 px\n"
                "Y: %13 px\n\n"

                "Centre Offset:\n"
                "X: %14 px\n"
                "Y: %15 px\n\n"

                "Output:\n%16"
            )
            .arg(detection.confidence * 100.0f, 0, 'f', 2)
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
            .arg(imageCenterX, 0, 'f', 1)
            .arg(imageCenterY, 0, 'f', 1)
            .arg(offsetX, 0, 'f', 1)
            .arg(offsetY, 0, 'f', 1)
            .arg(outputPath)
        );
    }
    catch (const cv::Exception& exception)
    {
        ui.lblstatus->setText(
            "Status : Hybrid OpenCV Error"
        );

        QMessageBox::critical(
            this,
            "OpenCV Error",
            QString::fromLocal8Bit(exception.what())
        );
    }
    catch (const std::exception& exception)
    {
        ui.lblstatus->setText(
            "Status : Hybrid Processing Error"
        );

        QMessageBox::critical(
            this,
            "Processing Error",
            QString::fromLocal8Bit(exception.what())
        );
    }
    catch (...)
    {
        ui.lblstatus->setText(
            "Status : Hybrid Processing Failed"
        );

        QMessageBox::critical(
            this,
            "Processing Error",
            "An unknown error occurred."
        );
    }
}



void AutoCentering::autoCenterImage()
{
    // ========================================================
    // DETECTION RESULT STRUCTURE
    // ========================================================

    struct DetectionInfo
    {
        bool found = false;

        cv::Point2f center;

        cv::RotatedRect box;

        float confidence = -1.0f;

        QString message;
    };

    // ========================================================
    // RESET FUNCTION
    // ========================================================

    auto setCenteringFailed = [this]()
        {
            ui.lblstatus->setText("Status : Failed");

            ui.labeloffsetx->setText(
                "Center Offset X : --"
            );

            ui.labeloffsety->setText(
                "Center Offset Y : --"
            );

            ui.labelconfidence->setText(
                "Confidence : --"
            );
        };

    // ========================================================
    // CHECK IMAGE
    // ========================================================

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

    // ========================================================
    // CHECK SELECTED METHOD
    // ========================================================

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

    const DetectionMethod selectedMethod =
        activeMethod;

    // ========================================================
    // METHOD NAME
    // ========================================================

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

    // ========================================================
    // CHECK YOLO MODEL
    // ========================================================

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

    // ========================================================
    // STOP PREVIOUS ANIMATION
    // ========================================================

    if (shiftAnimation)
    {
        shiftAnimation->stop();
        shiftAnimation->deleteLater();
        shiftAnimation = nullptr;
    }

    ui.lblstatus->setText(
        "Status : Detecting Component..."
    );

    ui.labeloffsetx->setText(
        "Center Offset X : Processing..."
    );

    ui.labeloffsety->setText(
        "Center Offset Y : Processing..."
    );

    ui.labelconfidence->setText(
        "Confidence : Processing..."
    );

    // ========================================================
    // CONVERT ORIGINAL IMAGE
    // ========================================================

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

    // ========================================================
    // CREATE DIAGNOSTIC DIRECTORY
    // ========================================================

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

        diagnosticDirectory =
            diagnosticFolder.toStdString();
    }

    // ========================================================
    // COMMON DETECTION FUNCTION
    //
    // Supports:
    // 1. YOLO
    // 2. Geometric Refinement
    // 3. YOLO + Geometric Refinement
    // ========================================================

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

            // ====================================================
            // METHOD 1: YOLO
            // ====================================================

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

            // ====================================================
            // METHOD 2: GEOMETRIC REFINEMENT
            // ====================================================

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

                info.message =
                    "Geometric Refinement successful.";

                return info;
            }

            // ====================================================
            // METHOD 3: YOLO + GEOMETRIC REFINEMENT
            // ====================================================

            if (selectedMethod ==
                DetectionMethod::YOLOGeometricRefinement)
            {
                // STEP 1: YOLO DETECTION

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

                // STEP 2: CREATE YOLO ROTATED BOUNDING BOX

                const float angleDegrees =
                    detection.angleRadians * 180.0f /
                    static_cast<float>(CV_PI);

                const cv::RotatedRect yoloBox(
                    detection.center,
                    detection.size,
                    angleDegrees
                );

                // STEP 3: YOLO-GUIDED OPENCV REFINEMENT

                YOLOGuidedRefinementResult result =
                    OpenCVOnlyGeometry::refineWithinYOLORegion(
                        inputImage,
                        yoloBox,
                        diagnosticPath,
                        0.25f
                    );

                if (!result.success ||
                    !result.refinedContourFound)
                {
                    info.message =
                        QString::fromStdString(result.message);

                    return info;
                }

                // STEP 4: GET REFINED BOUNDING BOX

                const cv::RotatedRect refinedBox =
                    result.refinedRotatedBoundingBox;

                if (refinedBox.size.width <= 0.0f ||
                    refinedBox.size.height <= 0.0f)
                {
                    info.message =
                        "Invalid refined bounding box.";

                    return info;
                }

                // STEP 5: RETURN REFINED GEOMETRY

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

    // ========================================================
    // INITIAL DETECTION
    // ========================================================

    DetectionInfo initialDetection;

    try
    {
        initialDetection = detectObject(
            original,
            diagnosticDirectory
        );
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

        ui.lblImageStatus->setText(
            initialDetection.message
        );

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

    // ========================================================
    // CALCULATE ROTATION CORRECTION
    // ========================================================

    const double correctionAngle =
        calculateRotationCorrection(
            initialDetection.box
        );

    // ========================================================
    // ROTATE IMAGE WITHOUT CROPPING
    // ========================================================

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
            std::ceil(
                original.rows * sine +
                original.cols * cosine
            )
            )
    );

    const int rotatedHeight = std::max(
        1,
        static_cast<int>(
            std::ceil(
                original.rows * cosine +
                original.cols * sine
            )
            )
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

    // ========================================================
    // DETECT COMPONENT AFTER ROTATION
    // ========================================================

    DetectionInfo rotatedDetection;

    try
    {
        rotatedDetection = detectObject(
            rotated,
            diagnosticDirectory
        );
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

    // ========================================================
    // CALCULATE TRANSLATION
    // ========================================================

    const double shiftX =
        outputCenter.x - rotatedDetection.center.x;

    const double shiftY =
        outputCenter.y - rotatedDetection.center.y;

    // ========================================================
    // COMBINE ROTATION AND TRANSLATION
    // ========================================================

    cv::Mat finalTransform =
        rotationMatrix.clone();

    finalTransform.at<double>(0, 2) += shiftX;

    finalTransform.at<double>(1, 2) += shiftY;

    // ========================================================
    // CREATE CORRECTED IMAGE
    // ========================================================

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

    // ========================================================
    // FINAL DETECTION
    // ========================================================

    DetectionInfo finalDetection;

    try
    {
        finalDetection = detectObject(
            corrected,
            diagnosticDirectory
        );
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

    // ========================================================
    // CREATE FINAL PREVIEW
    // ========================================================

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
        cv::Point(
            originalWidth / 2,
            originalHeight / 2
        ),
        cv::Scalar(255, 0, 0),
        cv::MARKER_CROSS,
        25,
        2,
        cv::LINE_AA
    );

    // ========================================================
    // SAVE OUTPUT
    // ========================================================

    const QString outputFolder =
        QDir::currentPath() +
        "/output/auto_centering";

    if (!QDir().mkpath(outputFolder))
    {
        setCenteringFailed();

        QMessageBox::warning(
            this,
            "Output Error",
            "Could not create the output folder."
        );

        return;
    }

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

    const QString correctedPath =
        QDir(outputFolder).filePath(
            baseName +
            methodSuffix +
            "_corrected.png"
        );

    const QString previewPath =
        QDir(outputFolder).filePath(
            baseName +
            methodSuffix +
            "_preview.png"
        );

    try
    {
        if (!cv::imwrite(
            correctedPath.toStdString(),
            corrected))
        {
            setCenteringFailed();

            QMessageBox::warning(
                this,
                "Save Error",
                "Could not save the corrected image."
            );

            return;
        }

        if (!cv::imwrite(
            previewPath.toStdString(),
            preview))
        {
            setCenteringFailed();

            QMessageBox::warning(
                this,
                "Save Error",
                "Could not save the preview image."
            );

            return;
        }
    }
    catch (const cv::Exception& exception)
    {
        setCenteringFailed();

        QMessageBox::critical(
            this,
            "Save Error",
            QString::fromLocal8Bit(exception.what())
        );

        return;
    }

    // ========================================================
    // PREPARE FINAL DISPLAY
    // ========================================================

    const QImage finalDisplayImage =
        bgrToQImage(preview);

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

    // ========================================================
    // PREPARE ANIMATION DATA
    // ========================================================

    const cv::Mat animationImage =
        original.clone();

    const cv::RotatedRect animationBox =
        initialDetection.box;

    const cv::Point2f animationCenter =
        initialDetection.center;

    const cv::Mat animationTransform =
        finalTransform.clone();

    // ========================================================
    // UPDATE UI
    // ========================================================

    if (initialDetection.confidence >= 0.0f)
    {
        ui.labelconfidence->setText(
            QString("Confidence : %1 %")
            .arg(
                initialDetection.confidence * 100.0f,
                0,
                'f',
                2
            )
        );
    }
    else
    {
        ui.labelconfidence->setText(
            "Confidence : N/A (OpenCV)"
        );
    }

    ui.lblstatus->setText(
        "Status : Starting Auto Center..."
    );

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

    // ========================================================
    // START ANIMATION
    // ========================================================

    shiftAnimation = new QVariantAnimation(this);

    shiftAnimation->setDuration(4000);

    shiftAnimation->setStartValue(0.0);

    shiftAnimation->setEndValue(1.0);

    shiftAnimation->setEasingCurve(
        QEasingCurve::InOutSine
    );

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
            const double progress =
                value.toDouble();

            // Interpolate affine transformation.

            cv::Mat currentTransform =
                cv::Mat::eye(2, 3, CV_64F);

            for (int row = 0; row < 2; ++row)
            {
                for (int column = 0; column < 3; ++column)
                {
                    const double startValue =
                        row == column ? 1.0 : 0.0;

                    const double finalValue =
                        animationTransform.at<double>(
                            row,
                            column
                        );

                    currentTransform.at<double>(
                        row,
                        column
                    ) =
                        startValue +
                        (finalValue - startValue) * progress;
                }
            }

            // Transform image.

            cv::Mat frame;

            cv::warpAffine(
                animationImage,
                frame,
                currentTransform,
                cv::Size(
                    originalWidth,
                    originalHeight
                ),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(255, 255, 255)
            );

            // Transform bounding box.

            cv::Point2f corners[4];

            animationBox.points(corners);

            for (int i = 0; i < 4; ++i)
            {
                const cv::Point2f point =
                    corners[i];

                corners[i] = cv::Point2f(
                    static_cast<float>(
                        currentTransform.at<double>(0, 0) *
                        point.x +
                        currentTransform.at<double>(0, 1) *
                        point.y +
                        currentTransform.at<double>(0, 2)
                        ),
                    static_cast<float>(
                        currentTransform.at<double>(1, 0) *
                        point.x +
                        currentTransform.at<double>(1, 1) *
                        point.y +
                        currentTransform.at<double>(1, 2)
                        )
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
                    currentTransform.at<double>(0, 0) *
                    animationCenter.x +
                    currentTransform.at<double>(0, 1) *
                    animationCenter.y +
                    currentTransform.at<double>(0, 2)
                    ),
                static_cast<float>(
                    currentTransform.at<double>(1, 0) *
                    animationCenter.x +
                    currentTransform.at<double>(1, 1) *
                    animationCenter.y +
                    currentTransform.at<double>(1, 2)
                    )
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
                cv::Point(
                    originalWidth / 2,
                    originalHeight / 2
                ),
                cv::Scalar(255, 0, 0),
                cv::MARKER_CROSS,
                25,
                2,
                cv::LINE_AA
            );

            // Display animation.

            displayImage(
                ui.originalImageLabel,
                bgrToQImage(frame)
            );

            // Remaining offset.

            const double remainingX =
                movingCenter.x - outputCenter.x;

            const double remainingY =
                movingCenter.y - outputCenter.y;

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
                    .arg(
                        static_cast<int>(progress * 100.0)
                    )
                );
            }
        }
    );

    // ========================================================
    // ANIMATION COMPLETION
    // ========================================================

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
            methodName,
            correctedPath,
            previewPath
        ]()
        {
            displayImage(
                ui.originalImageLabel,
                finalDisplayImage
            );

            const double finalOffsetX =
                finalDetection.center.x -
                originalWidth / 2.0;

            const double finalOffsetY =
                finalDetection.center.y -
                originalHeight / 2.0;

            ui.lblstatus->setText(
                "Status : Completed"
            );

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
                    .arg(
                        finalDetection.confidence * 100.0f,
                        0,
                        'f',
                        2
                    )
                );
            }
            else
            {
                ui.labelconfidence->setText(
                    "Confidence : N/A (OpenCV)"
                );
            }

            ui.lblImageStatus->setText(
                QString(
                    "%1 Auto Center completed.\n"
                    "Output dimensions: %2 x %3 px\n"
                    "Rotation correction: %4 degrees\n"
                    "Final offset X: %5 px\n"
                    "Final offset Y: %6 px\n\n"
                    "Corrected image:\n%7\n\n"
                    "Preview image:\n%8"
                )
                .arg(methodName)
                .arg(originalWidth)
                .arg(originalHeight)
                .arg(correctionAngle, 0, 'f', 2)
                .arg(finalOffsetX, 0, 'f', 2)
                .arg(finalOffsetY, 0, 'f', 2)
                .arg(correctedPath)
                .arg(previewPath)
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


void AutoCentering::startBatchAutoMode()
{
    // ========================================================
    // 1. CHECK INPUT FOLDER
    // ========================================================

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

    // ========================================================
    // 2. CHECK DETECTION METHOD
    // ========================================================

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

    // ========================================================
    // 3. CHECK YOLO MODEL
    // ========================================================

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

    // ========================================================
    // 4. FIND SUPPORTED IMAGES
    // ========================================================

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

    // ========================================================
    // 5. CREATE FULL IMAGE PATHS
    // ========================================================

    QStringList imagePaths;

    for (const QString& fileName : imageFiles)
    {
        imagePaths.append(
            inputFolder.filePath(fileName)
        );
    }

    // ========================================================
    // 6. PREPARE OUTPUT DIRECTORY
    // ========================================================

    const QString outputDirectory =
        QDir::currentPath() +
        "/output/batch_auto_centering";

    if (!QDir().mkpath(outputDirectory))
    {
        QMessageBox::critical(
            this,
            "Output Error",
            "Could not create the batch output folder."
        );

        return;
    }

    // ========================================================
    // 7. START BATCH PROCESSING
    // ========================================================

    btnStartBatch->setEnabled(false);

    processBatchImages(
        imagePaths,
        outputDirectory,
        selectedMethod
    );

    btnStartBatch->setEnabled(true);
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
        // ----------------------------------------------------
        // 1. CHECK INPUT IMAGE
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // 2. PREPARE DIAGNOSTIC DIRECTORIES
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // 3. DETECTION RESULT STRUCTURE
        // ----------------------------------------------------

        struct DetectionInfo
        {
            bool found = false;

            cv::Point2f center;

            cv::RotatedRect box;

            float confidence = -1.0f;

            QString message;
        };

        // ----------------------------------------------------
        // 4. COMMON DETECTION FUNCTION
        // ----------------------------------------------------

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

                // ================================================
                // METHOD 1: YOLO
                // ================================================

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

                // ================================================
                // METHOD 2: OPENCV GEOMETRIC REFINEMENT
                // ================================================

                if (selectedMethod ==
                    DetectionMethod::GeometricRefinement)
                {
                    OpenCVOnlySegmentationResult result =
                        OpenCVOnlyGeometry::segmentFullImage(
                            image,
                            diagnostic
                        );

                    if (!result.success ||
                        !result.outerContourFound)
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

                    info.message =
                        "Geometric Refinement successful.";

                    return info;
                }

                // ================================================
                // METHOD 3: YOLO + GEOMETRIC REFINEMENT
                // ================================================

                if (selectedMethod ==
                    DetectionMethod::YOLOGeometricRefinement)
                {
                    if (!detector || !detector->isLoaded())
                    {
                        info.message = "YOLO model is not loaded.";
                        return info;
                    }

                    // Step 1: YOLO detection.

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

                    // Step 2: YOLO-guided OpenCV refinement.

                    YOLOGuidedRefinementResult result =
                        OpenCVOnlyGeometry::refineWithinYOLORegion(
                            image,
                            yoloBox,
                            diagnostic,
                            0.25f
                        );

                    if (!result.success ||
                        !result.refinedContourFound)
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
                        info.message =
                            "Invalid refined bounding box.";

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

        // ----------------------------------------------------
        // 5. INITIAL DETECTION
        // ----------------------------------------------------

        const DetectionInfo initialDetection =
            detectObject(
                original,
                initialDiagnostic
            );

        if (!initialDetection.found)
        {
            output.errorMessage =
                "Initial detection failed: " +
                initialDetection.message;

            return output;
        }

        // Store initial detection confidence.

        output.confidence =
            initialDetection.confidence;

        // ----------------------------------------------------
        // 6. CALCULATE ROTATION CORRECTION
        // ----------------------------------------------------

        const double correctionAngle =
            calculateRotationCorrection(
                initialDetection.box
            );

        output.rotationCorrection =
            correctionAngle;

        // ----------------------------------------------------
        // 7. ROTATE IMAGE WITHOUT CROPPING
        // ----------------------------------------------------

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
                std::ceil(
                    original.rows * sine +
                    original.cols * cosine
                )
                )
        );

        const int rotatedHeight = std::max(
            1,
            static_cast<int>(
                std::ceil(
                    original.rows * cosine +
                    original.cols * sine
                )
                )
        );

        rotationMatrix.at<double>(0, 2) +=
            rotatedWidth / 2.0 - originalCenter.x;

        rotationMatrix.at<double>(1, 2) +=
            rotatedHeight / 2.0 - originalCenter.y;

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

        // ----------------------------------------------------
        // 8. DETECT AFTER ROTATION
        // ----------------------------------------------------

        const DetectionInfo rotatedDetection =
            detectObject(
                rotated,
                rotatedDiagnostic
            );

        if (!rotatedDetection.found)
        {
            output.errorMessage =
                "Detection after rotation failed: " +
                rotatedDetection.message;

            return output;
        }

        // ----------------------------------------------------
        // 9. CALCULATE TRANSLATION
        // ----------------------------------------------------

        const double shiftX =
            outputCenter.x - rotatedDetection.center.x;

        const double shiftY =
            outputCenter.y - rotatedDetection.center.y;

        output.translationX = shiftX;

        output.translationY = shiftY;

        // ----------------------------------------------------
        // 10. COMBINE ROTATION AND TRANSLATION
        // ----------------------------------------------------

        cv::Mat finalTransform =
            rotationMatrix.clone();

        finalTransform.at<double>(0, 2) += shiftX;

        finalTransform.at<double>(1, 2) += shiftY;

        // ----------------------------------------------------
        // 11. CREATE CORRECTED IMAGE
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // 12. FINAL DETECTION
        // ----------------------------------------------------

        const DetectionInfo finalDetection =
            detectObject(
                corrected,
                finalDiagnostic
            );

        if (!finalDetection.found)
        {
            output.errorMessage =
                "Final detection failed: " +
                finalDetection.message;

            return output;
        }

        // ----------------------------------------------------
        // 13. CALCULATE FINAL RESIDUAL OFFSET
        // ----------------------------------------------------

        output.finalOffsetX =
            finalDetection.center.x - outputCenter.x;

        output.finalOffsetY =
            finalDetection.center.y - outputCenter.y;

        // ----------------------------------------------------
        // 14. CREATE FINAL PREVIEW
        // ----------------------------------------------------

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
            cv::Point(
                originalWidth / 2,
                originalHeight / 2
            ),
            cv::Scalar(255, 0, 0),
            cv::MARKER_CROSS,
            25,
            2,
            cv::LINE_AA
        );

        // ----------------------------------------------------
        // 15. RETURN RESULTS
        // ----------------------------------------------------

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
