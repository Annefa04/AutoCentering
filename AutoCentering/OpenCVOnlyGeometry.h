#pragma once

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

// ============================================================
// OPENCV-ONLY SEGMENTATION RESULT
// ============================================================

struct OpenCVOnlySegmentationResult
{
    bool success = false;
    bool outerContourFound = false;

    std::string message;

    cv::Mat grayImage;
    cv::Mat enhancedImage;

    cv::Mat otsuNormalMask;
    cv::Mat otsuInverseMask;

    cv::Mat adaptiveNormalMask;
    cv::Mat adaptiveInverseMask;
    cv::Mat adaptiveInverseC2Mask;

    cv::Mat combinedInverseMask;
    cv::Mat closedCombinedInverseMask;

    cv::Mat outerCableMask;

    std::vector<cv::Point> outerCableContour;

    cv::Rect outerBoundingBox;

    cv::RotatedRect outerRotatedBoundingBox;

    cv::Mat allContoursImage;
    cv::Mat candidateVisualization;

    int contourCount = 0;
};

// ============================================================
// YOLO-GUIDED GEOMETRIC REFINEMENT RESULT
// ============================================================

struct YOLOGuidedRefinementResult
{
    bool success = false;
    bool refinedContourFound = false;

    std::string message;

    // ROI extracted using YOLO detection
    cv::Rect roi;

    // Refined geometry in FULL IMAGE coordinates
    std::vector<cv::Point> refinedContour;

    cv::Rect refinedBoundingBox;

    cv::RotatedRect refinedRotatedBoundingBox;

    // Visual results
    cv::Mat roiVisualization;
    cv::Mat fullImageVisualization;

    // Original OpenCV segmentation diagnostics.
    // These coordinates remain ROI-local.
    OpenCVOnlySegmentationResult segmentation;
};

// ============================================================
// OPENCV GEOMETRY CLASS
// ============================================================

class OpenCVOnlyGeometry
{
public:

    // Existing OpenCV-only method.
    // Keep this method unchanged.
    static OpenCVOnlySegmentationResult segmentFullImage(
        const cv::Mat& bgrImage,
        const std::string& outputDirectory = ""
    );

    // New hybrid method:
    // YOLO bounding box -> ROI -> OpenCV refinement.
    static YOLOGuidedRefinementResult refineWithinYOLORegion(
        const cv::Mat& bgrImage,
        const cv::RotatedRect& yoloBoundingBox,
        const std::string& outputDirectory = "",
        float paddingRatio = 0.25f
    );
};