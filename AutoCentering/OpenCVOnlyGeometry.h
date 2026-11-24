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

    // NOTE: if maxImageSide is used these images/masks are at the
    // WORKING (downscaled) resolution. Contours/boxes/outerCableMask
    // are always in ORIGINAL image coordinates.
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

    // time spent in segmentFullImage (milliseconds)
    double processingTimeMs = 0.0;
};

// ============================================================
// YOLO-GUIDED GEOMETRIC REFINEMENT RESULT
// ============================================================

struct YOLOGuidedRefinementResult
{
    bool success = false;
    bool refinedContourFound = false;

    std::string message;

    // ROI extracted using YOLO detection (full image coordinates)
    cv::Rect roi;

    // Refined geometry in FULL IMAGE coordinates
    std::vector<cv::Point> refinedContour;

    cv::Rect refinedBoundingBox;

    cv::RotatedRect refinedRotatedBoundingBox;

    // Only filled when RefinementOptions::buildVisualizations = true
    cv::Mat roiVisualization;
    cv::Mat fullImageVisualization;

    // Geometry is ROI-local (original ROI scale). Masks/images are only
    // filled when buildVisualizations = true, at working resolution.
    OpenCVOnlySegmentationResult segmentation;

    // time spent in refineWithinYOLORegion (milliseconds)
    double processingTimeMs = 0.0;
};

// ============================================================
// OPTIONS - GEOMETRIC (OpenCV-only) METHOD
// Defaults reproduce the original results.
// ============================================================

struct SegmentationOptions
{
    bool saveDiagnostics = false;   // write images/report to outputDirectory (slow)
    bool buildVisualizations = false;   // fill candidateVisualization / allContoursImage / combined masks
    std::string outputDirectory;

    int  maxImageSide = 0;              // 0 = no downscale (identical to original). Try 1280 / 1024
    bool needAllMasks = true;           // false: Otsu Inverse + Adaptive Inverse C2 only (faster)
    std::vector<int> kernelSizes = { 3, 5, 7 };   // try { 3, 5 } for speed
    bool parallel = true;               // run mask/kernel combinations on several cores
};

// ============================================================
// OPTIONS - HYBRID (YOLO + Geometric) METHOD
// Defaults reproduce the original results.
// ============================================================

struct RefinementOptions
{
    bool saveDiagnostics = false;
    bool buildVisualizations = false;
    std::string outputDirectory;

    float paddingRatio = 0.25f;         // original value. Try 0.15
    int   maxRoiSide = 0;             // 0 = no downscale. Try 640

    bool needAllMasks = true;           // false: Otsu Inverse + Adaptive Inverse C2 only
    std::vector<int> kernelSizes = { 3, 5, 7 };   // try { 3, 5 }
    bool parallel = true;
};

// ============================================================
// OPENCV GEOMETRY CLASS
// ============================================================

class OpenCVOnlyGeometry
{
public:

    // Fast Geometric method with options.
    static OpenCVOnlySegmentationResult segmentFullImage(
        const cv::Mat& bgrImage,
        const SegmentationOptions& options
    );

    // Old signature (same behaviour as before: visualizations ON,
    // diagnostics saved only if outputDirectory is not empty).
    static OpenCVOnlySegmentationResult segmentFullImage(
        const cv::Mat& bgrImage,
        const std::string& outputDirectory = ""
    );

    // Fast hybrid method with options.
    static YOLOGuidedRefinementResult refineWithinYOLORegion(
        const cv::Mat& bgrImage,
        const cv::RotatedRect& yoloBoundingBox,
        const RefinementOptions& options
    );

    // Old signature (diagnostics/visualizations ON only if outputDirectory is not empty).
    static YOLOGuidedRefinementResult refineWithinYOLORegion(
        const cv::Mat& bgrImage,
        const cv::RotatedRect& yoloBoundingBox,
        const std::string& outputDirectory = "",
        float paddingRatio = 0.25f
    );
};