#include "OpenCVOnlyGeometry.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace
{
    // ============================================================
    // SAVE DIAGNOSTIC IMAGE
    // ============================================================

    void saveDiagnostic(
        const std::string& directory,
        const std::string& filename,
        const cv::Mat& image)
    {
        if (directory.empty() || image.empty())
            return;

        try
        {
            std::string path = directory;

            if (path.back() != '/' && path.back() != '\\')
                path += '/';

            path += filename;

            cv::imwrite(path, image);
        }
        catch (...)
        {
            // Diagnostic saving must not interrupt processing.
        }
    }

    // ============================================================
    // DRAW ROTATED BOUNDING BOX
    // ============================================================

    void drawRotatedBox(
        cv::Mat& image,
        const cv::RotatedRect& rotatedBox,
        const cv::Scalar& colour = cv::Scalar(255, 0, 0),
        int thickness = 2)
    {
        cv::Point2f corners[4];

        rotatedBox.points(corners);

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

    // ============================================================
    // COMPONENT CANDIDATE
    // ============================================================

    struct ComponentCandidate
    {
        std::vector<cv::Point> contour;

        cv::Rect boundingBox;
        cv::RotatedRect rotatedBox;

        double area = 0.0;
        double aspectRatio = 0.0;
        double solidity = 0.0;
        double extent = 0.0;
        double score = 0.0;
    };

    // ============================================================
    // EVALUATE COMPONENT FOR OPENCV-ONLY DETECTION
    // ============================================================

    bool evaluateComponent(
        const std::vector<cv::Point>& contour,
        cv::Size imageSize,
        ComponentCandidate& candidate)
    {
        if (contour.size() < 5)
            return false;

        const double imageArea =
            static_cast<double>(imageSize.width) *
            imageSize.height;

        const double area =
            cv::contourArea(contour);

        if (area < imageArea * 0.00005)
            return false;

        if (area > imageArea * 0.60)
            return false;

        const cv::RotatedRect rotatedBox =
            cv::minAreaRect(contour);

        const double width = rotatedBox.size.width;
        const double height = rotatedBox.size.height;

        if (width < 5.0 || height < 5.0)
            return false;

        const double longerSide =
            std::max(width, height);

        const double shorterSide =
            std::min(width, height);

        if (shorterSide <= 0.0)
            return false;

        const double aspectRatio =
            longerSide / shorterSide;

        if (aspectRatio > 4.5)
            return false;

        const cv::Rect box =
            cv::boundingRect(contour);

        if (box.width < 5 || box.height < 5)
            return false;

        const double boxArea =
            static_cast<double>(box.width) * box.height;

        if (boxArea <= 0.0)
            return false;

        const double extent = area / boxArea;

        std::vector<cv::Point> hull;

        cv::convexHull(contour, hull);

        const double hullArea =
            cv::contourArea(hull);

        if (hullArea <= 0.0)
            return false;

        const double solidity =
            area / hullArea;

        if (solidity < 0.35)
            return false;

        candidate.contour = contour;
        candidate.boundingBox = box;
        candidate.rotatedBox = rotatedBox;
        candidate.area = area;
        candidate.aspectRatio = aspectRatio;
        candidate.solidity = solidity;
        candidate.extent = extent;

        candidate.score =
            area *
            (0.5 + solidity) *
            (0.5 + extent) /
            (1.0 + aspectRatio * 0.15);

        return true;
    }

    // ============================================================
    // FIND MAIN COMPONENT FOR OPENCV-ONLY METHOD
    // ============================================================

    bool findMainComponentContour(
        const std::vector<cv::Mat>& thresholdMasks,
        cv::Size imageSize,
        std::vector<cv::Point>& selectedContour,
        cv::Mat& selectedMask,
        cv::Rect& selectedBoundingBox,
        cv::RotatedRect& selectedRotatedBox,
        int& totalContours)
    {
        double bestScore = -1.0;

        totalContours = 0;

        const int kernelSizes[] = { 3, 5, 7 };

        for (const cv::Mat& sourceMask : thresholdMasks)
        {
            if (sourceMask.empty())
                continue;

            for (int kernelSize : kernelSizes)
            {
                cv::Mat processedMask;

                const cv::Mat kernel =
                    cv::getStructuringElement(
                        cv::MORPH_ELLIPSE,
                        cv::Size(kernelSize, kernelSize)
                    );

                cv::morphologyEx(
                    sourceMask,
                    processedMask,
                    cv::MORPH_CLOSE,
                    kernel
                );

                std::vector<std::vector<cv::Point>> contours;

                cv::findContours(
                    processedMask,
                    contours,
                    cv::RETR_EXTERNAL,
                    cv::CHAIN_APPROX_SIMPLE
                );

                totalContours +=
                    static_cast<int>(contours.size());

                for (const auto& contour : contours)
                {
                    ComponentCandidate candidate;

                    if (!evaluateComponent(
                        contour,
                        imageSize,
                        candidate))
                    {
                        continue;
                    }

                    if (candidate.score > bestScore)
                    {
                        bestScore = candidate.score;

                        selectedContour = candidate.contour;
                        selectedBoundingBox = candidate.boundingBox;
                        selectedRotatedBox = candidate.rotatedBox;

                        selectedMask =
                            cv::Mat::zeros(
                                imageSize,
                                CV_8UC1
                            );

                        std::vector<std::vector<cv::Point>> selected =
                        {
                            selectedContour
                        };

                        cv::drawContours(
                            selectedMask,
                            selected,
                            0,
                            cv::Scalar(255),
                            cv::FILLED
                        );
                    }
                }
            }
        }

        return !selectedContour.empty();
    }

    // ============================================================
    // YOLO-GUIDED CONTOUR CANDIDATE
    // ============================================================

    struct YOLOGuidedCandidate
    {
        std::vector<cv::Point> contour;

        cv::Rect boundingBox;
        cv::RotatedRect rotatedBox;

        double area = 0.0;
        double overlapRatio = 0.0;
        double yoloCoverage = 0.0;
        double iou = 0.0;
        double centreScore = 0.0;
        double areaScore = 0.0;
        double solidity = 0.0;
        double aspectRatio = 0.0;

        double finalScore = -1.0;
    };

    // ============================================================
    // SELECT BEST CONTOUR USING YOLO AS REFERENCE
    // ============================================================

    bool findBestYOLOGuidedContour(
        const std::vector<cv::Mat>& masks,
        const cv::Mat& yoloMask,
        const cv::RotatedRect& localYOLOBox,
        std::vector<cv::Point>& bestContour,
        cv::Mat& bestContourMask,
        int& totalContours)
    {
        totalContours = 0;

        double bestScore = -1.0;

        const double yoloArea =
            static_cast<double>(
                cv::countNonZero(yoloMask)
                );

        if (yoloArea <= 0.0)
            return false;

        const double referenceDiagonal =
            std::sqrt(
                localYOLOBox.size.width *
                localYOLOBox.size.width +
                localYOLOBox.size.height *
                localYOLOBox.size.height
            );

        if (referenceDiagonal <= 0.0)
            return false;

        const int kernelSizes[] = { 3, 5, 7 };

        for (const cv::Mat& sourceMask : masks)
        {
            if (sourceMask.empty())
                continue;

            if (sourceMask.size() != yoloMask.size())
                continue;

            for (int kernelSize : kernelSizes)
            {
                cv::Mat processedMask;

                const cv::Mat kernel =
                    cv::getStructuringElement(
                        cv::MORPH_ELLIPSE,
                        cv::Size(kernelSize, kernelSize)
                    );

                cv::morphologyEx(
                    sourceMask,
                    processedMask,
                    cv::MORPH_CLOSE,
                    kernel
                );

                std::vector<std::vector<cv::Point>> contours;

                cv::findContours(
                    processedMask,
                    contours,
                    cv::RETR_EXTERNAL,
                    cv::CHAIN_APPROX_SIMPLE
                );

                totalContours +=
                    static_cast<int>(contours.size());

                for (const auto& contour : contours)
                {
                    if (contour.size() < 5)
                        continue;

                    const double contourArea =
                        cv::contourArea(contour);

                    // Reject very small noise.
                    if (contourArea < yoloArea * 0.05)
                        continue;

                    // Reject candidates that are excessively
                    // larger than the YOLO reference region.
                    if (contourArea > yoloArea * 2.50)
                        continue;

                    const cv::RotatedRect rotatedBox =
                        cv::minAreaRect(contour);

                    const double width =
                        rotatedBox.size.width;

                    const double height =
                        rotatedBox.size.height;

                    if (width < 5.0 || height < 5.0)
                        continue;

                    const double longerSide =
                        std::max(width, height);

                    const double shorterSide =
                        std::min(width, height);

                    if (shorterSide <= 0.0)
                        continue;

                    const double aspectRatio =
                        longerSide / shorterSide;

                    // Reject very thin wire-like contours.
                    // This is deliberately more permissive than
                    // the OpenCV-only method.
                    if (aspectRatio > 10.0)
                        continue;

                    // ----------------------------------------
                    // CREATE CANDIDATE MASK
                    // ----------------------------------------

                    cv::Mat candidateMask =
                        cv::Mat::zeros(
                            yoloMask.size(),
                            CV_8UC1
                        );

                    std::vector<std::vector<cv::Point>> candidateContours =
                    {
                        contour
                    };

                    cv::drawContours(
                        candidateMask,
                        candidateContours,
                        0,
                        cv::Scalar(255),
                        cv::FILLED
                    );

                    const double candidatePixelArea =
                        cv::countNonZero(candidateMask);

                    if (candidatePixelArea <= 0.0)
                        continue;

                    // ----------------------------------------
                    // YOLO / CONTOUR INTERSECTION
                    // ----------------------------------------

                    cv::Mat intersectionMask;

                    cv::bitwise_and(
                        candidateMask,
                        yoloMask,
                        intersectionMask
                    );

                    const double intersectionArea =
                        cv::countNonZero(intersectionMask);

                    const double unionArea =
                        candidatePixelArea +
                        yoloArea -
                        intersectionArea;

                    if (unionArea <= 0.0)
                        continue;

                    // Percentage of candidate inside YOLO.
                    const double overlapRatio =
                        intersectionArea /
                        candidatePixelArea;

                    // Percentage of YOLO region covered by candidate.
                    const double yoloCoverage =
                        intersectionArea /
                        yoloArea;

                    const double iou =
                        intersectionArea /
                        unionArea;

                    // Reject candidates that barely overlap YOLO.
                    if (overlapRatio < 0.30)
                        continue;

                    if (yoloCoverage < 0.20)
                        continue;

                    // ----------------------------------------
                    // CENTRE DISTANCE
                    // ----------------------------------------

                    const cv::Moments moments =
                        cv::moments(contour);

                    if (std::abs(moments.m00) < 1e-6)
                        continue;

                    const cv::Point2f candidateCentre(
                        static_cast<float>(
                            moments.m10 / moments.m00
                            ),
                        static_cast<float>(
                            moments.m01 / moments.m00
                            )
                    );

                    const double centreDistance =
                        cv::norm(
                            candidateCentre -
                            localYOLOBox.center
                        );

                    const double normalizedDistance =
                        centreDistance /
                        referenceDiagonal;

                    const double centreScore =
                        std::exp(
                            -2.0 * normalizedDistance
                        );

                    // ----------------------------------------
                    // AREA SIMILARITY
                    // ----------------------------------------

                    const double areaRatio =
                        candidatePixelArea / yoloArea;

                    const double areaScore =
                        std::exp(
                            -0.65 *
                            std::abs(
                                std::log(
                                    std::max(areaRatio, 1e-6)
                                )
                            )
                        );

                    // ----------------------------------------
                    // CONTOUR SOLIDITY
                    // ----------------------------------------

                    std::vector<cv::Point> hull;

                    cv::convexHull(contour, hull);

                    const double hullArea =
                        cv::contourArea(hull);

                    if (hullArea <= 0.0)
                        continue;

                    const double solidity =
                        contourArea / hullArea;

                    if (solidity < 0.30)
                        continue;

                    // ----------------------------------------
                    // FINAL SCORE
                    // ----------------------------------------

                    // YOLO overlap is important, but the score
                    // also rewards contours that cover the
                    // detected component and have a plausible
                    // centre and area.

                    const double finalScore =
                        (0.25 * overlapRatio) +
                        (0.25 * yoloCoverage) +
                        (0.20 * iou) +
                        (0.15 * centreScore) +
                        (0.10 * areaScore) +
                        (0.05 * solidity);

                    if (finalScore > bestScore)
                    {
                        bestScore = finalScore;

                        bestContour = contour;

                        bestContourMask =
                            candidateMask.clone();
                    }
                }
            }
        }

        return !bestContour.empty();
    }

    // ============================================================
    // CONVERT IMAGE TO GRAYSCALE
    // ============================================================

    bool convertToGray(
        const cv::Mat& input,
        cv::Mat& gray)
    {
        if (input.empty() || input.depth() != CV_8U)
            return false;

        if (input.channels() == 1)
        {
            gray = input.clone();
        }
        else if (input.channels() == 3)
        {
            cv::cvtColor(
                input,
                gray,
                cv::COLOR_BGR2GRAY
            );
        }
        else if (input.channels() == 4)
        {
            cv::cvtColor(
                input,
                gray,
                cv::COLOR_BGRA2GRAY
            );
        }
        else
        {
            return false;
        }

        return !gray.empty();
    }
}

// ============================================================
// OPENCV-ONLY COMPONENT SEGMENTATION
// ============================================================

OpenCVOnlySegmentationResult
OpenCVOnlyGeometry::segmentFullImage(
    const cv::Mat& bgrImage,
    const std::string& outputDirectory)
{
    OpenCVOnlySegmentationResult result;

    try
    {
        // ----------------------------------------------------
        // 1. VALIDATE IMAGE
        // ----------------------------------------------------

        if (bgrImage.empty())
        {
            result.message = "Input image is empty.";
            return result;
        }

        if (bgrImage.cols < 32 || bgrImage.rows < 32)
        {
            result.message = "Input image is too small.";
            return result;
        }

        // ----------------------------------------------------
        // 2. GRAYSCALE
        // ----------------------------------------------------

        if (!convertToGray(bgrImage, result.grayImage))
        {
            result.message = "Unsupported image format.";
            return result;
        }

        saveDiagnostic(
            outputDirectory,
            "01_grayscale.png",
            result.grayImage
        );

        // ----------------------------------------------------
        // 3. CONTRAST ENHANCEMENT
        // ----------------------------------------------------

        cv::Ptr<cv::CLAHE> clahe =
            cv::createCLAHE(
                2.0,
                cv::Size(8, 8)
            );

        clahe->apply(
            result.grayImage,
            result.enhancedImage
        );

        cv::GaussianBlur(
            result.enhancedImage,
            result.enhancedImage,
            cv::Size(3, 3),
            0.8
        );

        saveDiagnostic(
            outputDirectory,
            "02_enhanced.png",
            result.enhancedImage
        );

        // ----------------------------------------------------
        // 4. OTSU NORMAL
        // ----------------------------------------------------

        cv::threshold(
            result.enhancedImage,
            result.otsuNormalMask,
            0,
            255,
            cv::THRESH_BINARY | cv::THRESH_OTSU
        );

        saveDiagnostic(
            outputDirectory,
            "03_otsu_normal.png",
            result.otsuNormalMask
        );

        // ----------------------------------------------------
        // 5. OTSU INVERSE
        // ----------------------------------------------------

        cv::threshold(
            result.enhancedImage,
            result.otsuInverseMask,
            0,
            255,
            cv::THRESH_BINARY_INV | cv::THRESH_OTSU
        );

        saveDiagnostic(
            outputDirectory,
            "04_otsu_inverse.png",
            result.otsuInverseMask
        );

        // ----------------------------------------------------
        // 6. ADAPTIVE THRESHOLD
        // ----------------------------------------------------

        int blockSize = std::min(
            31,
            std::min(
                result.enhancedImage.rows,
                result.enhancedImage.cols
            )
        );

        if (blockSize % 2 == 0)
            blockSize--;

        blockSize = std::max(3, blockSize);

        cv::adaptiveThreshold(
            result.enhancedImage,
            result.adaptiveNormalMask,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY,
            blockSize,
            4
        );

        cv::adaptiveThreshold(
            result.enhancedImage,
            result.adaptiveInverseMask,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,
            blockSize,
            4
        );

        cv::adaptiveThreshold(
            result.enhancedImage,
            result.adaptiveInverseC2Mask,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,
            blockSize,
            2
        );

        saveDiagnostic(
            outputDirectory,
            "05_adaptive_normal.png",
            result.adaptiveNormalMask
        );

        saveDiagnostic(
            outputDirectory,
            "06_adaptive_inverse.png",
            result.adaptiveInverseMask
        );

        saveDiagnostic(
            outputDirectory,
            "07_adaptive_inverse_c2.png",
            result.adaptiveInverseC2Mask
        );

        // ----------------------------------------------------
        // 7. COMBINE INVERSE MASKS
        // ----------------------------------------------------

        cv::bitwise_or(
            result.otsuInverseMask,
            result.adaptiveInverseC2Mask,
            result.combinedInverseMask
        );

        const cv::Mat smallKernel =
            cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(3, 3)
            );

        cv::morphologyEx(
            result.combinedInverseMask,
            result.closedCombinedInverseMask,
            cv::MORPH_CLOSE,
            smallKernel
        );

        saveDiagnostic(
            outputDirectory,
            "08_combined_inverse.png",
            result.combinedInverseMask
        );

        saveDiagnostic(
            outputDirectory,
            "09_combined_closed.png",
            result.closedCombinedInverseMask
        );

        // ----------------------------------------------------
        // 8. FIND MAIN COMPONENT
        // ----------------------------------------------------

        const std::vector<cv::Mat> masks =
        {
            result.otsuNormalMask,
            result.otsuInverseMask,
            result.adaptiveNormalMask,
            result.adaptiveInverseMask,
            result.adaptiveInverseC2Mask
        };

        result.outerContourFound =
            findMainComponentContour(
                masks,
                result.grayImage.size(),
                result.outerCableContour,
                result.outerCableMask,
                result.outerBoundingBox,
                result.outerRotatedBoundingBox,
                result.contourCount
            );

        // ----------------------------------------------------
        // 9. VISUALIZATION
        // ----------------------------------------------------

        if (bgrImage.channels() == 1)
        {
            cv::cvtColor(
                bgrImage,
                result.candidateVisualization,
                cv::COLOR_GRAY2BGR
            );

            result.allContoursImage =
                result.candidateVisualization.clone();
        }
        else if (bgrImage.channels() == 4)
        {
            cv::cvtColor(
                bgrImage,
                result.candidateVisualization,
                cv::COLOR_BGRA2BGR
            );

            result.allContoursImage =
                result.candidateVisualization.clone();
        }
        else
        {
            result.candidateVisualization = bgrImage.clone();
            result.allContoursImage = bgrImage.clone();
        }

        if (result.outerContourFound)
        {
            const std::vector<std::vector<cv::Point>> contours =
            {
                result.outerCableContour
            };

            // Red: component outer contour.
            cv::drawContours(
                result.candidateVisualization,
                contours,
                0,
                cv::Scalar(0, 0, 255),
                2,
                cv::LINE_AA
            );

            // Blue: rotated bounding box.
            drawRotatedBox(
                result.candidateVisualization,
                result.outerRotatedBoundingBox
            );

            // Green: component centre.
            cv::circle(
                result.candidateVisualization,
                result.outerRotatedBoundingBox.center,
                5,
                cv::Scalar(0, 255, 0),
                cv::FILLED
            );

            cv::drawContours(
                result.allContoursImage,
                contours,
                0,
                cv::Scalar(0, 0, 255),
                2,
                cv::LINE_AA
            );

            saveDiagnostic(
                outputDirectory,
                "10_component_mask.png",
                result.outerCableMask
            );

            result.message =
                "Main component candidate detected successfully.";
        }
        else
        {
            result.message =
                "No suitable component body was detected.";
        }

        saveDiagnostic(
            outputDirectory,
            "11_component_result.png",
            result.candidateVisualization
        );

        saveDiagnostic(
            outputDirectory,
            "12_component_contour.png",
            result.allContoursImage
        );

        // ----------------------------------------------------
        // 10. COMPONENT INFORMATION
        // ----------------------------------------------------

        if (!outputDirectory.empty() &&
            result.outerContourFound)
        {
            std::string reportPath = outputDirectory;

            if (reportPath.back() != '/' &&
                reportPath.back() != '\\')
            {
                reportPath += '/';
            }

            reportPath += "13_component_information.txt";

            std::ofstream report(reportPath);

            if (report.is_open())
            {
                report << "OPENCV COMPONENT DETECTION\n\n";

                report << "Image width: "
                    << bgrImage.cols << '\n';

                report << "Image height: "
                    << bgrImage.rows << '\n';

                report << "Contours examined: "
                    << result.contourCount << '\n';

                report << "Contour points: "
                    << result.outerCableContour.size() << '\n';

                report << "Bounding box X: "
                    << result.outerBoundingBox.x << '\n';

                report << "Bounding box Y: "
                    << result.outerBoundingBox.y << '\n';

                report << "Bounding box width: "
                    << result.outerBoundingBox.width << '\n';

                report << "Bounding box height: "
                    << result.outerBoundingBox.height << '\n';

                report << "Centre X: "
                    << result.outerRotatedBoundingBox.center.x << '\n';

                report << "Centre Y: "
                    << result.outerRotatedBoundingBox.center.y << '\n';

                report << "Rotation angle: "
                    << result.outerRotatedBoundingBox.angle << '\n';

                report << "Rotated width: "
                    << result.outerRotatedBoundingBox.size.width << '\n';

                report << "Rotated height: "
                    << result.outerRotatedBoundingBox.size.height << '\n';
            }
        }

        result.success = true;

        return result;
    }
    catch (const cv::Exception& e)
    {
        result.success = false;
        result.outerContourFound = false;

        result.message =
            std::string("OpenCV exception: ") + e.what();

        return result;
    }
    catch (const std::exception& e)
    {
        result.success = false;
        result.outerContourFound = false;

        result.message =
            std::string("Processing exception: ") + e.what();

        return result;
    }
    catch (...)
    {
        result.success = false;
        result.outerContourFound = false;

        result.message =
            "Unknown error during OpenCV segmentation.";

        return result;
    }
}

// ============================================================
// YOLO + GEOMETRIC REFINEMENT
// ============================================================

YOLOGuidedRefinementResult
OpenCVOnlyGeometry::refineWithinYOLORegion(
    const cv::Mat& bgrImage,
    const cv::RotatedRect& yoloBoundingBox,
    const std::string& outputDirectory,
    float paddingRatio)
{
    YOLOGuidedRefinementResult result;

    try
    {
        // ----------------------------------------------------
        // 1. VALIDATE IMAGE
        // ----------------------------------------------------

        if (bgrImage.empty())
        {
            result.message = "Input image is empty.";
            return result;
        }

        if (bgrImage.depth() != CV_8U)
        {
            result.message = "Image must have 8-bit pixel depth.";
            return result;
        }

        if (bgrImage.cols < 32 || bgrImage.rows < 32)
        {
            result.message = "Input image is too small.";
            return result;
        }

        cv::Mat displaySource;

        if (bgrImage.channels() == 1)
        {
            cv::cvtColor(
                bgrImage,
                displaySource,
                cv::COLOR_GRAY2BGR
            );
        }
        else if (bgrImage.channels() == 3)
        {
            displaySource = bgrImage.clone();
        }
        else if (bgrImage.channels() == 4)
        {
            cv::cvtColor(
                bgrImage,
                displaySource,
                cv::COLOR_BGRA2BGR
            );
        }
        else
        {
            result.message = "Unsupported image format.";
            return result;
        }

        // ----------------------------------------------------
        // 2. VALIDATE YOLO BOX
        // ----------------------------------------------------

        if (!std::isfinite(yoloBoundingBox.center.x) ||
            !std::isfinite(yoloBoundingBox.center.y) ||
            !std::isfinite(yoloBoundingBox.size.width) ||
            !std::isfinite(yoloBoundingBox.size.height) ||
            !std::isfinite(yoloBoundingBox.angle) ||
            yoloBoundingBox.size.width <= 0.0f ||
            yoloBoundingBox.size.height <= 0.0f)
        {
            result.message = "Invalid YOLO bounding box.";
            return result;
        }

        // ----------------------------------------------------
        // 3. GET YOLO BOX CORNERS
        // ----------------------------------------------------

        cv::Point2f yoloCorners[4];

        yoloBoundingBox.points(yoloCorners);

        std::vector<cv::Point2f> yoloCornerVector(
            yoloCorners,
            yoloCorners + 4
        );

        cv::Rect yoloRectangle =
            cv::boundingRect(yoloCornerVector);

        if (yoloRectangle.width <= 0 ||
            yoloRectangle.height <= 0)
        {
            result.message = "Could not calculate YOLO region.";
            return result;
        }

        // ----------------------------------------------------
        // 4. EXPAND YOLO REGION
        // ----------------------------------------------------

        paddingRatio =
            std::max(0.0f, std::min(0.75f, paddingRatio));

        const int paddingX =
            std::max(
                12,
                static_cast<int>(
                    std::ceil(
                        yoloRectangle.width * paddingRatio
                    )
                    )
            );

        const int paddingY =
            std::max(
                12,
                static_cast<int>(
                    std::ceil(
                        yoloRectangle.height * paddingRatio
                    )
                    )
            );

        const cv::Rect expandedROI(
            yoloRectangle.x - paddingX,
            yoloRectangle.y - paddingY,
            yoloRectangle.width + 2 * paddingX,
            yoloRectangle.height + 2 * paddingY
        );

        // ----------------------------------------------------
        // 5. CLIP ROI TO IMAGE
        // ----------------------------------------------------

        const cv::Rect imageBounds(
            0,
            0,
            bgrImage.cols,
            bgrImage.rows
        );

        result.roi = expandedROI & imageBounds;

        if (result.roi.width < 32 ||
            result.roi.height < 32)
        {
            result.message = "YOLO ROI is too small.";
            return result;
        }

        // ----------------------------------------------------
        // 6. EXTRACT YOLO ROI
        // ----------------------------------------------------

        const cv::Mat roiImage =
            displaySource(result.roi).clone();

        if (roiImage.empty())
        {
            result.message = "Failed to extract YOLO region.";
            return result;
        }

        // ----------------------------------------------------
        // 7. GENERATE SEGMENTATION MASKS INSIDE ROI
        // ----------------------------------------------------

        result.segmentation =
            segmentFullImage(
                roiImage,
                outputDirectory
            );

        if (!result.segmentation.success)
        {
            result.message =
                "ROI segmentation failed: " +
                result.segmentation.message;

            return result;
        }

        // ----------------------------------------------------
        // 8. CONVERT YOLO BOX TO ROI COORDINATES
        // ----------------------------------------------------

        cv::RotatedRect localYOLOBox =
            yoloBoundingBox;

        localYOLOBox.center.x -=
            static_cast<float>(result.roi.x);

        localYOLOBox.center.y -=
            static_cast<float>(result.roi.y);

        // ----------------------------------------------------
        // 9. CREATE YOLO REFERENCE MASK
        // ----------------------------------------------------

        cv::Point2f localCorners[4];

        localYOLOBox.points(localCorners);

        std::vector<cv::Point> localPolygon;

        for (int i = 0; i < 4; ++i)
        {
            localPolygon.emplace_back(
                cvRound(localCorners[i].x),
                cvRound(localCorners[i].y)
            );
        }

        cv::Mat yoloMask =
            cv::Mat::zeros(
                roiImage.size(),
                CV_8UC1
            );

        const std::vector<std::vector<cv::Point>> yoloPolygons =
        {
            localPolygon
        };

        cv::fillPoly(
            yoloMask,
            yoloPolygons,
            cv::Scalar(255)
        );

        // ----------------------------------------------------
        // 10. COLLECT THRESHOLD MASKS
        // ----------------------------------------------------

        const std::vector<cv::Mat> candidateMasks =
        {
            result.segmentation.otsuNormalMask,
            result.segmentation.otsuInverseMask,
            result.segmentation.adaptiveNormalMask,
            result.segmentation.adaptiveInverseMask,
            result.segmentation.adaptiveInverseC2Mask
        };

        // ----------------------------------------------------
        // 11. FIND BEST YOLO-GUIDED CONTOUR
        // ----------------------------------------------------

        std::vector<cv::Point> selectedContour;

        cv::Mat selectedMask;

        int totalContours = 0;

        const bool found =
            findBestYOLOGuidedContour(
                candidateMasks,
                yoloMask,
                localYOLOBox,
                selectedContour,
                selectedMask,
                totalContours
            );

        result.segmentation.contourCount =
            totalContours;

        if (!found)
        {
            result.message =
                "YOLO detected the component, but OpenCV could "
                "not find a suitable matching contour. "
                "Inspect the ROI threshold images.";

            return result;
        }

        // ----------------------------------------------------
        // 12. REFINE CONTOUR
        // ----------------------------------------------------

        // Use a small approximation tolerance to preserve
        // the actual component boundary.

        std::vector<cv::Point> refinedContour;

        const double perimeter =
            cv::arcLength(
                selectedContour,
                true
            );

        const double epsilon =
            std::max(
                0.5,
                perimeter * 0.0005
            );

        cv::approxPolyDP(
            selectedContour,
            refinedContour,
            epsilon,
            true
        );

        if (refinedContour.size() < 5)
        {
            refinedContour = selectedContour;
        }

        // ----------------------------------------------------
        // 13. CALCULATE FINAL REFINED GEOMETRY
        // ----------------------------------------------------

        // The final rotated bounding box follows the selected
        // OpenCV contour instead of the original YOLO rectangle.

        const cv::RotatedRect refinedLocalBox =
            cv::minAreaRect(refinedContour);

        const cv::Rect refinedLocalBoundingBox =
            cv::boundingRect(refinedContour);

        // Update ROI-local segmentation results.

        result.segmentation.outerContourFound = true;

        result.segmentation.outerCableContour =
            refinedContour;

        result.segmentation.outerBoundingBox =
            refinedLocalBoundingBox;

        result.segmentation.outerRotatedBoundingBox =
            refinedLocalBox;

        result.segmentation.outerCableMask =
            cv::Mat::zeros(
                roiImage.size(),
                CV_8UC1
            );

        const std::vector<std::vector<cv::Point>> refinedLocalContours =
        {
            refinedContour
        };

        cv::drawContours(
            result.segmentation.outerCableMask,
            refinedLocalContours,
            0,
            cv::Scalar(255),
            cv::FILLED
        );

        // ----------------------------------------------------
        // 14. CONVERT CONTOUR TO FULL IMAGE COORDINATES
        // ----------------------------------------------------

        result.refinedContour =
            refinedContour;

        for (cv::Point& point : result.refinedContour)
        {
            point.x += result.roi.x;
            point.y += result.roi.y;
        }

        result.refinedBoundingBox =
            refinedLocalBoundingBox;

        result.refinedBoundingBox.x += result.roi.x;
        result.refinedBoundingBox.y += result.roi.y;

        result.refinedRotatedBoundingBox =
            refinedLocalBox;

        result.refinedRotatedBoundingBox.center.x +=
            static_cast<float>(result.roi.x);

        result.refinedRotatedBoundingBox.center.y +=
            static_cast<float>(result.roi.y);

        // ----------------------------------------------------
        // 15. ROI VISUALIZATION
        // ----------------------------------------------------

        result.roiVisualization =
            roiImage.clone();

        // Yellow: YOLO reference bounding box.
        drawRotatedBox(
            result.roiVisualization,
            localYOLOBox,
            cv::Scalar(0, 255, 255),
            2
        );

        // Red: refined component contour.
        cv::drawContours(
            result.roiVisualization,
            refinedLocalContours,
            0,
            cv::Scalar(0, 0, 255),
            2,
            cv::LINE_AA
        );

        // Blue: refined rotated bounding box.
        drawRotatedBox(
            result.roiVisualization,
            refinedLocalBox,
            cv::Scalar(255, 0, 0),
            2
        );

        // Green: refined centre.
        cv::circle(
            result.roiVisualization,
            refinedLocalBox.center,
            5,
            cv::Scalar(0, 255, 0),
            cv::FILLED
        );

        saveDiagnostic(
            outputDirectory,
            "14_yolo_guided_refinement_roi.png",
            result.roiVisualization
        );

        // ----------------------------------------------------
        // 16. FULL IMAGE VISUALIZATION
        // ----------------------------------------------------

        result.fullImageVisualization =
            displaySource.clone();

        // Yellow: original YOLO bounding box.
        drawRotatedBox(
            result.fullImageVisualization,
            yoloBoundingBox,
            cv::Scalar(0, 255, 255),
            2
        );

        // Red: refined component contour.
        const std::vector<std::vector<cv::Point>> fullContours =
        {
            result.refinedContour
        };

        cv::drawContours(
            result.fullImageVisualization,
            fullContours,
            0,
            cv::Scalar(0, 0, 255),
            2,
            cv::LINE_AA
        );

        // Blue: final refined bounding box.
        drawRotatedBox(
            result.fullImageVisualization,
            result.refinedRotatedBoundingBox,
            cv::Scalar(255, 0, 0),
            2
        );

        // Green: final centre.
        cv::circle(
            result.fullImageVisualization,
            result.refinedRotatedBoundingBox.center,
            6,
            cv::Scalar(0, 255, 0),
            cv::FILLED
        );

        saveDiagnostic(
            outputDirectory,
            "15_yolo_geometric_final_result.png",
            result.fullImageVisualization
        );

        // ----------------------------------------------------
        // 17. SAVE COMPONENT INFORMATION
        // ----------------------------------------------------

        if (!outputDirectory.empty())
        {
            std::string reportPath = outputDirectory;

            if (reportPath.back() != '/' &&
                reportPath.back() != '\\')
            {
                reportPath += '/';
            }

            reportPath +=
                "16_yolo_geometric_information.txt";

            std::ofstream report(reportPath);

            if (report.is_open())
            {
                report << "YOLO + GEOMETRIC REFINEMENT\n\n";

                report << "YOLO centre X: "
                    << yoloBoundingBox.center.x << '\n';

                report << "YOLO centre Y: "
                    << yoloBoundingBox.center.y << '\n';

                report << "YOLO width: "
                    << yoloBoundingBox.size.width << '\n';

                report << "YOLO height: "
                    << yoloBoundingBox.size.height << '\n';

                report << "YOLO angle: "
                    << yoloBoundingBox.angle << '\n';

                report << "\nREFINED GEOMETRY\n\n";

                report << "Refined contour points: "
                    << result.refinedContour.size() << '\n';

                report << "Refined centre X: "
                    << result.refinedRotatedBoundingBox.center.x
                    << '\n';

                report << "Refined centre Y: "
                    << result.refinedRotatedBoundingBox.center.y
                    << '\n';

                report << "Refined width: "
                    << result.refinedRotatedBoundingBox.size.width
                    << '\n';

                report << "Refined height: "
                    << result.refinedRotatedBoundingBox.size.height
                    << '\n';

                report << "Refined angle: "
                    << result.refinedRotatedBoundingBox.angle
                    << '\n';

                report << "Contours examined: "
                    << totalContours << '\n';
            }
        }

        // ----------------------------------------------------
        // 18. FINISH
        // ----------------------------------------------------

        result.refinedContourFound = true;
        result.success = true;

        result.message =
            "YOLO detection and geometric refinement completed.";

        return result;
    }
    catch (const cv::Exception& e)
    {
        result.success = false;
        result.refinedContourFound = false;

        result.message =
            std::string("OpenCV exception: ") + e.what();

        return result;
    }
    catch (const std::exception& e)
    {
        result.success = false;
        result.refinedContourFound = false;

        result.message =
            std::string("Processing exception: ") + e.what();

        return result;
    }
    catch (...)
    {
        result.success = false;
        result.refinedContourFound = false;

        result.message =
            "Unknown error during YOLO geometric refinement.";

        return result;
    }
}