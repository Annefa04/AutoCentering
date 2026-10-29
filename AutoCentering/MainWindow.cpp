
#include "MainWindow.h"

#include <QFileDialog>
#include <QPixmap>
#include <QMessageBox>
#include <QPainter>
#include <QPen>



#include <opencv2/opencv.hpp>

#include <array>
#include <algorithm>
#include <cmath>
#include <vector>
#include <QDir>
#include <QFileInfo>

#include <QEasingCurve>
#include <QVariantAnimation>
#include <QCheckBox>



AutoCentering::AutoCentering(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);
    // Minimalist Auto Mode toggle styling
    ui.chkAutoCenterOnLoad->setChecked(false);
    ui.chkAutoCenterOnLoad->setCursor(Qt::PointingHandCursor);

    ui.chkAutoCenterOnLoad->setStyleSheet(R"(
        QCheckBox#chkAutoCenterOnLoad {
            color: #64748B;
            spacing: 8px;
            font-size: 11px;
            font-weight: 600;
            padding: 4px;
        }

        QCheckBox#chkAutoCenterOnLoad::indicator {
            width: 36px;
            height: 20px;
            border-radius: 10px;
            border: 1px solid #CBD5E1;
            background-color: #CBD5E1;
        }

        QCheckBox#chkAutoCenterOnLoad::indicator:hover {
            border-color: #94A3B8;
        }

        QCheckBox#chkAutoCenterOnLoad::indicator:checked {
            background-color: #22C55E;
            border: 1px solid #16A34A;
        }
    )");

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
        ui.btnAutoCenter,
        &QPushButton::clicked,
        this,
        &AutoCentering::autoCenterImage
    );
    // Auto Mode ON/OFF
    connect(
        ui.chkAutoCenterOnLoad,
        &QCheckBox::toggled,
        this,
        [this](bool enabled)
        {
            if (enabled && !currentImage.isNull())
            {
                autoCenterImage();
            }
            else if (!enabled)
            {
                ui.lblImageStatus->setText(
                    "Auto Mode OFF. Manual controls are available."
                );
            }
        }
    );
}

AutoCentering::~AutoCentering()
{
    delete detector;
}

void AutoCentering::loadImage()
{

    QString fileName = QFileDialog::getOpenFileName(
        this,
        "Open Image",
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
            "Error",
            "Failed to load image."
        );
        return;
    }

    currentImagePath = fileName;
    currentImage = loadedImage;

    ui.originalImageLabel->setPixmap(
        QPixmap::fromImage(currentImage).scaled(
            ui.originalImageLabel->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        )
    );

    ui.lblImageStatus->setText("Image Loaded Successfully");
    ui.lblstatus->setText("Status : Waiting");
    ui.labeloffsetx->setText("Center Offset X : 0 px");
    ui.labeloffsety->setText("Center Offset Y : 0 px");
    ui.labelconfidence->setText("Confidence : 0 %");
}

void AutoCentering::processImage()
{
    if (currentImage.isNull())
    {
        QMessageBox::warning(
            this,
            "Warning",
            "Please load an image first."
        );
        return;
    }

    if (!detector || !detector->isLoaded())
    {
        QMessageBox::critical(
            this,
            "YOLO",
            "The ONNX model is not loaded."
        );
        return;
    }

    QImage rgb = currentImage.convertToFormat(
        QImage::Format_RGB888
    );

    cv::Mat rgbMat(
        rgb.height(),
        rgb.width(),
        CV_8UC3,
        rgb.bits(),
        static_cast<size_t>(rgb.bytesPerLine())
    );

    cv::Mat bgr;
    cv::cvtColor(rgbMat, bgr, cv::COLOR_RGB2BGR);

    const CableDetection detection = detector->detect(bgr);

    if (!detection.found)
    {
        ui.lblImageStatus->setText(
            "No cable detected."
        );

        QMessageBox::information(
            this,
            "Detection",
            "No detection passed the confidence threshold."
        );
        return;
    }

    QImage resultImage = currentImage.convertToFormat(
        QImage::Format_RGB32
    );

    QPainter painter(&resultImage);
    painter.setRenderHint(QPainter::Antialiasing);

    QPen pen(Qt::green);
    pen.setWidth(4);
    painter.setPen(pen);

    const float angleDegrees =
        detection.angleRadians * 180.0f /
        static_cast<float>(CV_PI);

    cv::RotatedRect box(
        detection.center,
        detection.size,
        angleDegrees
    );

    cv::Point2f vertices[4];
    box.points(vertices);

    for (int i = 0; i < 4; ++i)
    {
        painter.drawLine(
            QPointF(vertices[i].x, vertices[i].y),
            QPointF(
                vertices[(i + 1) % 4].x,
                vertices[(i + 1) % 4].y
            )
        );
    }

    painter.setBrush(Qt::red);
    painter.drawEllipse(
        QPointF(detection.center.x, detection.center.y),
        5,
        5
    );

    painter.end();

    ui.originalImageLabel->setPixmap(
        QPixmap::fromImage(resultImage).scaled(
            ui.originalImageLabel->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        )
    );

    const float imageCenterX = currentImage.width() / 2.0f;
    const float imageCenterY = currentImage.height() / 2.0f;

    const float offsetX = detection.center.x - imageCenterX;
    const float offsetY = detection.center.y - imageCenterY;

    ui.lblImageStatus->setText(
        QString(
            "Cable detected\n"
            "Confidence: %1%\n"
            "Cable center: (%2, %3)\n"
            "Image center: (%4, %5)\n"
            "Offset: X %6 px, Y %7 px\n"
            "Angle: %8 degrees"
        )
        .arg(detection.confidence * 100.0f, 0, 'f', 2)
        .arg(detection.center.x, 0, 'f', 1)
        .arg(detection.center.y, 0, 'f', 1)
        .arg(imageCenterX, 0, 'f', 1)
        .arg(imageCenterY, 0, 'f', 1)
        .arg(offsetX, 0, 'f', 1)
        .arg(offsetY, 0, 'f', 1)
        .arg(angleDegrees, 0, 'f', 2)
    );
}




void AutoCentering::autoCenterImage()
{
    // ========================================================
    // STATUS HELPER
    // ========================================================

    auto setCenteringFailed = [this]()
        {
            ui.lblstatus->setText("Status : Failed");
            ui.labeloffsetx->setText("Center Offset X : --");
            ui.labeloffsety->setText("Center Offset Y : --");
            ui.labelconfidence->setText("Confidence : --");
        };

    // Reset previous results before starting.
    ui.lblstatus->setText("Status : Processing...");
    ui.labeloffsetx->setText("Center Offset X : --");
    ui.labeloffsety->setText("Center Offset Y : --");
    ui.labelconfidence->setText("Confidence : --");

    // ========================================================
    // 1. VALIDATE IMAGE AND YOLO MODEL
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

    if (!detector || !detector->isLoaded())
    {
        setCenteringFailed();

        QMessageBox::critical(
            this,
            "Auto Center",
            "The ONNX model is not loaded."
        );

        return;
    }

    const int originalWidth = currentImage.width();
    const int originalHeight = currentImage.height();

    // Always process the original image.
    QImage rgb = currentImage.convertToFormat(
        QImage::Format_RGB888
    );

    cv::Mat rgbMat(
        rgb.height(),
        rgb.width(),
        CV_8UC3,
        rgb.bits(),
        static_cast<size_t>(rgb.bytesPerLine())
    );

    cv::Mat original;

    cv::cvtColor(
        rgbMat,
        original,
        cv::COLOR_RGB2BGR
    );

    // ========================================================
    // 2. INITIAL YOLO OBB DETECTION
    // ========================================================

    const CableDetection initialDetection =
        detector->detect(original);

    if (!initialDetection.found)
    {
        setCenteringFailed();

        ui.lblImageStatus->setText(
            "Auto Center failed: no cable detected."
        );

        QMessageBox::information(
            this,
            "Auto Center",
            "No cable detected. The original image is unchanged."
        );

        return;
    }

    // ========================================================
    // 3. FIND THE LONGEST BOUNDING-BOX EDGE
    // ========================================================

    const float initialAngleDegrees =
        initialDetection.angleRadians * 180.0f /
        static_cast<float>(CV_PI);

    const cv::RotatedRect initialBox(
        initialDetection.center,
        initialDetection.size,
        initialAngleDegrees
    );

    cv::Point2f corners[4];

    initialBox.points(corners);

    double longestLength = 0.0;
    double longestEdgeAngle = 0.0;

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

            longestEdgeAngle =
                std::atan2(
                    static_cast<double>(edge.y),
                    static_cast<double>(edge.x)
                ) * 180.0 / CV_PI;
        }
    }

    if (longestLength < 1e-6)
    {
        setCenteringFailed();

        ui.lblImageStatus->setText(
            "Auto Center failed: invalid bounding box."
        );

        QMessageBox::warning(
            this,
            "Auto Center",
            "The detected bounding box is invalid."
        );

        return;
    }

    // Normalise the angle to [-90, 90).
    double normalizedAngle =
        std::fmod(longestEdgeAngle + 90.0, 180.0);

    if (normalizedAngle < 0.0)
        normalizedAngle += 180.0;

    normalizedAngle -= 90.0;

    // Snap to the closest horizontal or vertical axis.
    const double targetAngle =
        std::round(normalizedAngle / 90.0) * 90.0;

    double correctionAngle =
        normalizedAngle - targetAngle;

    while (correctionAngle > 90.0)
        correctionAngle -= 180.0;

    while (correctionAngle < -90.0)
        correctionAngle += 180.0;

    if (std::abs(correctionAngle) < 0.15)
        correctionAngle = 0.0;

    // ========================================================
    // 4. ROTATE THE IMAGE WITHOUT CROPPING
    // ========================================================

    cv::Mat rotated;

    if (std::abs(correctionAngle) < 1e-6)
    {
        rotated = original.clone();
    }
    else
    {
        const int width = original.cols;
        const int height = original.rows;

        const cv::Point2f imageCenter(
            width / 2.0f,
            height / 2.0f
        );

        cv::Mat rotationMatrix =
            cv::getRotationMatrix2D(
                imageCenter,
                correctionAngle,
                1.0
            );

        const double cosine =
            std::abs(rotationMatrix.at<double>(0, 0));

        const double sine =
            std::abs(rotationMatrix.at<double>(0, 1));

        const int newWidth = std::max(
            1,
            static_cast<int>(
                std::ceil(
                    height * sine + width * cosine
                )
                )
        );

        const int newHeight = std::max(
            1,
            static_cast<int>(
                std::ceil(
                    height * cosine + width * sine
                )
                )
        );

        rotationMatrix.at<double>(0, 2) +=
            newWidth / 2.0 - imageCenter.x;

        rotationMatrix.at<double>(1, 2) +=
            newHeight / 2.0 - imageCenter.y;

        cv::warpAffine(
            original,
            rotated,
            rotationMatrix,
            cv::Size(newWidth, newHeight),
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT,
            cv::Scalar(255, 255, 255)
        );
    }

    // ========================================================
    // 5. DETECT AGAIN AFTER ROTATION
    // ========================================================

    const CableDetection rotatedDetection =
        detector->detect(rotated);

    if (!rotatedDetection.found)
    {
        setCenteringFailed();

        ui.lblImageStatus->setText(
            "Auto Center failed: cable not detected after rotation."
        );

        QMessageBox::warning(
            this,
            "Auto Center",
            "The cable could not be detected after rotation."
        );

        return;
    }

    // ========================================================
    // 6. TRANSLATE THE CABLE TO THE IMAGE CENTRE
    // ========================================================

    const double rotatedCenterX = rotated.cols / 2.0;
    const double rotatedCenterY = rotated.rows / 2.0;

    const double shiftX =
        rotatedCenterX - rotatedDetection.center.x;

    const double shiftY =
        rotatedCenterY - rotatedDetection.center.y;

    // Add enough temporary padding for the translation.
    const int padX = static_cast<int>(
        std::ceil(std::abs(shiftX))
        );

    const int padY = static_cast<int>(
        std::ceil(std::abs(shiftY))
        );

    const int temporaryWidth =
        rotated.cols + 2 * padX;

    const int temporaryHeight =
        rotated.rows + 2 * padY;

    cv::Mat translationMatrix = (
        cv::Mat_<double>(2, 3) <<
        1.0, 0.0, shiftX + padX,
        0.0, 1.0, shiftY + padY
        );

    cv::Mat centeredTemporary;

    cv::warpAffine(
        rotated,
        centeredTemporary,
        translationMatrix,
        cv::Size(temporaryWidth, temporaryHeight),
        cv::INTER_CUBIC,
        cv::BORDER_CONSTANT,
        cv::Scalar(255, 255, 255)
    );

    // ========================================================
    // 7. CROP TO THE ORIGINAL IMAGE DIMENSIONS
    // ========================================================

    const int cropX =
        (centeredTemporary.cols - originalWidth) / 2;

    const int cropY =
        (centeredTemporary.rows - originalHeight) / 2;

    if (cropX < 0 ||
        cropY < 0 ||
        cropX + originalWidth > centeredTemporary.cols ||
        cropY + originalHeight > centeredTemporary.rows)
    {
        setCenteringFailed();

        ui.lblImageStatus->setText(
            "Auto Center failed: invalid crop area."
        );

        QMessageBox::critical(
            this,
            "Auto Center",
            "Could not crop the result to the original dimensions."
        );

        return;
    }

    const cv::Rect cropRegion(
        cropX,
        cropY,
        originalWidth,
        originalHeight
    );

    cv::Mat corrected =
        centeredTemporary(cropRegion).clone();

    // ========================================================
    // 8. FINAL YOLO DETECTION
    // ========================================================

    const CableDetection finalDetection =
        detector->detect(corrected);

    cv::Mat preview = corrected.clone();

    if (finalDetection.found)
    {
        const float finalAngleDegrees =
            finalDetection.angleRadians * 180.0f /
            static_cast<float>(CV_PI);

        const cv::RotatedRect finalBox(
            finalDetection.center,
            finalDetection.size,
            finalAngleDegrees
        );

        cv::Point2f finalCorners[4];

        finalBox.points(finalCorners);

        // Green rotated bounding box.
        for (int i = 0; i < 4; ++i)
        {
            cv::line(
                preview,
                finalCorners[i],
                finalCorners[(i + 1) % 4],
                cv::Scalar(0, 255, 0),
                3,
                cv::LINE_AA
            );
        }

        // Red circle: detected cable centre.
        cv::circle(
            preview,
            finalDetection.center,
            6,
            cv::Scalar(0, 0, 255),
            cv::FILLED,
            cv::LINE_AA
        );
    }

    // Blue cross: image centre.
    const cv::Point outputImageCenter(
        originalWidth / 2,
        originalHeight / 2
    );

    cv::drawMarker(
        preview,
        outputImageCenter,
        cv::Scalar(255, 0, 0),
        cv::MARKER_CROSS,
        25,
        2,
        cv::LINE_AA
    );

    // ========================================================
    // 9. SAVE CORRECTED IMAGE AND PREVIEW
    // ========================================================

    const QString outputFolder =
        QDir::currentPath() +
        "/output/auto_orientation_centering";

    if (!QDir().mkpath(outputFolder))
    {
        setCenteringFailed();

        ui.lblImageStatus->setText(
            "Auto Center failed: could not create output folder."
        );

        QMessageBox::warning(
            this,
            "Auto Center",
            "Could not create the output folder."
        );

        return;
    }

    QString baseName =
        QFileInfo(currentImagePath).completeBaseName();

    if (baseName.isEmpty())
        baseName = "xray_image";

    const QString correctedPath =
        QDir(outputFolder).filePath(
            baseName + "_corrected.png"
        );

    const QString previewPath =
        QDir(outputFolder).filePath(
            baseName + "_preview.png"
        );

    try
    {
        if (!cv::imwrite(
            correctedPath.toStdString(),
            corrected))
        {
            setCenteringFailed();

            ui.lblImageStatus->setText(
                "Auto Center failed: corrected image could not be saved."
            );

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

            ui.lblImageStatus->setText(
                "Auto Center failed: preview could not be saved."
            );

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
            QString("Failed to save output:\n%1")
            .arg(
                QString::fromLocal8Bit(exception.what())
            )
        );

        return;
    }


    // ========================================================
    // 10. SLOW ROTATION + TABLE-SHIFT SIMULATION
    // ========================================================

    // Prepare the actual final preview.
    cv::Mat previewRgb;
    cv::cvtColor(preview, previewRgb, cv::COLOR_BGR2RGB);

    QImage finalDisplayImage(
        previewRgb.data,
        previewRgb.cols,
        previewRgb.rows,
        static_cast<qsizetype>(previewRgb.step),
        QImage::Format_RGB888
    );

    finalDisplayImage = finalDisplayImage.copy();

    // Stop any previous animation.
    if (shiftAnimation)
    {
        shiftAnimation->stop();
        delete shiftAnimation;
        shiftAnimation = nullptr;
    }

    // Animate from the original image and its YOLO detection.
    const cv::Mat animationImage = original.clone();
    const CableDetection animationDetection = initialDetection;

    const cv::Point2f imageCenter(
        originalWidth / 2.0f,
        originalHeight / 2.0f
    );

    // Calculate the cable centre after the full rotation.
    cv::Mat fullRotation = cv::getRotationMatrix2D(
        imageCenter,
        correctionAngle,
        1.0
    );

    const cv::Point2f detectedCenter = animationDetection.center;

    const cv::Point2f rotatedCenter(
        static_cast<float>(
            fullRotation.at<double>(0, 0) * detectedCenter.x +
            fullRotation.at<double>(0, 1) * detectedCenter.y +
            fullRotation.at<double>(0, 2)
            ),
        static_cast<float>(
            fullRotation.at<double>(1, 0) * detectedCenter.x +
            fullRotation.at<double>(1, 1) * detectedCenter.y +
            fullRotation.at<double>(1, 2)
            )
    );

    // Target shift after rotation.
    const double targetShiftX =
        imageCenter.x - rotatedCenter.x;

    const double targetShiftY =
        imageCenter.y - rotatedCenter.y;

    const cv::RotatedRect animationBox(
        animationDetection.center,
        animationDetection.size,
        initialAngleDegrees
    );

    

    // Update Control Status before animation begins.
    ui.lblstatus->setText("Status : Starting simulation...");
    ui.labeloffsetx->setText("Center Offset X : Preparing...");
    ui.labeloffsety->setText("Center Offset Y : Preparing...");

    ui.labelconfidence->setText(
        QString("Confidence : %1 %").arg(
            animationDetection.confidence * 100.0f,
            0, 'f', 2
        )
    );

    ui.lblImageStatus->setText(
        "Simulating slow rotation and X-ray table movement..."
    );

    // 15 seconds total: 12 seconds rotation, then 3 seconds shift.
    constexpr double rotationPhase = 0.3;

    shiftAnimation = new QVariantAnimation(this);
    shiftAnimation->setDuration(4000);
    shiftAnimation->setStartValue(0.0);
    shiftAnimation->setEndValue(1.0);
    shiftAnimation->setEasingCurve(QEasingCurve::InOutSine);

    connect(
        shiftAnimation,
        &QVariantAnimation::valueChanged,
        this,
        [this,
        animationImage,
        animationDetection,
        animationBox,
        imageCenter,
        originalWidth,
        originalHeight,
        targetShiftX,
        targetShiftY,
        rotationPhase,
        correctionAngle](const QVariant& value)
        {
            const double progress = value.toDouble();

            // Phase 1: gradually rotate the cable.
            const double rotationProgress = std::clamp(
                progress / rotationPhase,
                0.0,
                1.0
            );

            const double currentAngle =
                correctionAngle * rotationProgress;

            // Phase 2: simulate the table shifting.
            const double shiftProgress = std::clamp(
                (progress - rotationPhase) /
                (1.0 - rotationPhase),
                0.0,
                1.0
            );

            const double currentShiftX =
                targetShiftX * shiftProgress;

            const double currentShiftY =
                targetShiftY * shiftProgress;

            // Apply rotation around the image centre.
            cv::Mat transform = cv::getRotationMatrix2D(
                imageCenter,
                currentAngle,
                1.0
            );

            // Apply the simulated translation.
            transform.at<double>(0, 2) += currentShiftX;
            transform.at<double>(1, 2) += currentShiftY;

            // Render the animation at the original image dimensions.
            cv::Mat frame;

            cv::warpAffine(
                animationImage,
                frame,
                transform,
                cv::Size(originalWidth, originalHeight),
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(255, 255, 255)
            );

            // Transform the detected bounding-box corners.
            cv::Point2f boxCorners[4];
            animationBox.points(boxCorners);

            for (int i = 0; i < 4; ++i)
            {
                const cv::Point2f p = boxCorners[i];

                boxCorners[i] = cv::Point2f(
                    static_cast<float>(
                        transform.at<double>(0, 0) * p.x +
                        transform.at<double>(0, 1) * p.y +
                        transform.at<double>(0, 2)
                        ),
                    static_cast<float>(
                        transform.at<double>(1, 0) * p.x +
                        transform.at<double>(1, 1) * p.y +
                        transform.at<double>(1, 2)
                        )
                );
            }

            // Green rotated bounding box.
            for (int i = 0; i < 4; ++i)
            {
                cv::line(
                    frame,
                    boxCorners[i],
                    boxCorners[(i + 1) % 4],
                    cv::Scalar(0, 255, 0),
                    3,
                    cv::LINE_AA
                );
            }

            // Red marker: transformed cable centre.
            const cv::Point2f movingCenter(
                static_cast<float>(
                    transform.at<double>(0, 0) *
                    animationDetection.center.x +
                    transform.at<double>(0, 1) *
                    animationDetection.center.y +
                    transform.at<double>(0, 2)
                    ),
                static_cast<float>(
                    transform.at<double>(1, 0) *
                    animationDetection.center.x +
                    transform.at<double>(1, 1) *
                    animationDetection.center.y +
                    transform.at<double>(1, 2)
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

            // Blue cross: fixed target image centre.
            const cv::Point targetCenter(
                originalWidth / 2,
                originalHeight / 2
            );

            cv::drawMarker(
                frame,
                targetCenter,
                cv::Scalar(255, 0, 0),
                cv::MARKER_CROSS,
                25,
                2,
                cv::LINE_AA
            );

            // Convert the frame to QImage.
            cv::Mat frameRgb;

            cv::cvtColor(
                frame,
                frameRgb,
                cv::COLOR_BGR2RGB
            );

            QImage frameImage(
                frameRgb.data,
                frameRgb.cols,
                frameRgb.rows,
                static_cast<qsizetype>(frameRgb.step),
                QImage::Format_RGB888
            );

            frameImage = frameImage.copy();

            ui.originalImageLabel->setPixmap(
                QPixmap::fromImage(frameImage).scaled(
                    ui.originalImageLabel->size(),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation
                )
            );

            // Update the remaining offsets during the animation.
            const double remainingX =
                movingCenter.x - targetCenter.x;

            const double remainingY =
                movingCenter.y - targetCenter.y;

            ui.labeloffsetx->setText(
                QString("Center Offset X : %1 px")
                .arg(remainingX, 0, 'f', 1)
            );

            ui.labeloffsety->setText(
                QString("Center Offset Y : %1 px")
                .arg(remainingY, 0, 'f', 1)
            );

            ui.lblstatus->setText(
                progress < rotationPhase
                ? "Status : Rotating cable slowly..."
                : "Status : Simulating table shift..."
            );
        }
    );

    // ========================================================
    // 11. DISPLAY FINAL RESULT AFTER THE ANIMATION
    // ========================================================

    connect(
        shiftAnimation,
        &QVariantAnimation::finished,
        this,
        [this,
        finalDisplayImage,
        finalDetection,
        originalWidth,
        originalHeight,
        correctionAngle]()
        {
            // Display the actual corrected preview.
            ui.originalImageLabel->setPixmap(
                QPixmap::fromImage(finalDisplayImage).scaled(
                    ui.originalImageLabel->size(),
                    Qt::KeepAspectRatio,
                    Qt::SmoothTransformation
                )
            );

            if (finalDetection.found)
            {
                const double finalOffsetX =
                    finalDetection.center.x - originalWidth / 2.0;

                const double finalOffsetY =
                    finalDetection.center.y - originalHeight / 2.0;

                ui.lblstatus->setText("Status : Completed");

                ui.labeloffsetx->setText(
                    QString("Center Offset X : %1 px")
                    .arg(finalOffsetX, 0, 'f', 1)
                );

                ui.labeloffsety->setText(
                    QString("Center Offset Y : %1 px")
                    .arg(finalOffsetY, 0, 'f', 1)
                );

                ui.labelconfidence->setText(
                    QString("Confidence : %1 %")
                    .arg(
                        finalDetection.confidence * 100.0f,
                        0,
                        'f',
                        2
                    )
                );

                ui.lblImageStatus->setText(
                    QString(
                        "Auto Center completed\n"
                        "Output size: %1 x %2 px\n"
                        "Rotation correction: %3 degrees\n"
                        "Corrected image and preview saved."
                    )
                    .arg(originalWidth)
                    .arg(originalHeight)
                    .arg(correctionAngle, 0, 'f', 2)
                );
            }
            else
            {
                ui.lblstatus->setText("Status : Failed");
                ui.labeloffsetx->setText("Center Offset X : --");
                ui.labeloffsety->setText("Center Offset Y : --");
                ui.labelconfidence->setText("Confidence : --");

                ui.lblImageStatus->setText(
                    "Animation finished, but final detection failed."
                );
            }

            // Release the completed animation.
            if (shiftAnimation)
            {
                shiftAnimation->deleteLater();
                shiftAnimation = nullptr;
            }
        }
    );

    // Start exactly one animation.
    shiftAnimation->start();
}

   





