#include "MainWindow.h"

#include <QFileDialog>
#include <QPixmap>
#include <QMessageBox>
#include <QStringList>

#include <opencv2/opencv.hpp>
#include <QPainter>
#include <QPen>


AutoCentering::AutoCentering(QWidget* parent)
    : QMainWindow(parent)
{
    ui.setupUi(this);

    detector = new YOLODetector(
        L"models/best.onnx"
    );

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

    currentImagePath = fileName;

    currentImage.load(fileName);

    if (currentImage.isNull())
    {
        QMessageBox::critical(
            this,
            "Error",
            "Failed to load image."
        );
        return;
    }

    QPixmap pixmap =
        QPixmap::fromImage(currentImage);

    ui.originalImageLabel->setPixmap(
        pixmap.scaled(
            ui.originalImageLabel->size(),
            Qt::KeepAspectRatio,
            Qt::SmoothTransformation
        )
    );

    ui.lblImageStatus->setText(
        "Image Loaded Successfully"
    );
}


void AutoCentering::processImage()
{
    if (currentImage.isNull())
    {
        QMessageBox::warning(this, "Warning",
            "Please load an image first.");
        return;
    }

    if (!detector || !detector->isLoaded())
    {
        QMessageBox::critical(this, "YOLO",
            "The ONNX model is not loaded.");
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
            "No cable detected. Check application output."
        );

        QMessageBox::information(
            this, "Detection",
            "No detection passed the confidence threshold."
        );
        return;
    }

    // Draw the rotated box over a copy of the original image.
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
            QPointF(vertices[(i + 1) % 4].x,
                vertices[(i + 1) % 4].y)
        );
    }

    painter.setBrush(Qt::red);
    painter.drawEllipse(
        QPointF(detection.center.x, detection.center.y),
        5, 5
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


