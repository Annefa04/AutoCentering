// ============================================================
// OpenCVOnlyGeometry.cpp  -  FAST VERSION (single file)
//
// Replace your whole OpenCVOnlyGeometry.cpp with this file and
// REMOVE OpenCVOnlyGeometry_Refine.cpp from the project (if added).
//
// Same selection logic / scoring as the original code, but:
//   - CLAHE and kernels created once
//   - no repeated full-image mask drawing
//   - candidate masks drawn only inside their bounding rect (hybrid)
//   - mask/kernel combinations run in parallel (cv::parallel_for_)
//   - optional downscale (maxImageSide / maxRoiSide)
//   - segmentFullImage is NOT called inside the hybrid any more
//   - no disk writes / visualizations unless requested
//
// Build in RELEASE (x64) and run with Ctrl+F5 for real timings.
// Requires C++17 (std::filesystem).
// ============================================================

#include "OpenCVOnlyGeometry.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

namespace
{
    // ============================================================
    // GENERAL HELPERS
    // ============================================================

    cv::Ptr<cv::CLAHE>& getClahe()
    {
        thread_local cv::Ptr<cv::CLAHE> clahe =
            cv::createCLAHE(2.0, cv::Size(8, 8));
        return clahe;
    }

    void saveFast(
        const std::string& directory,
        const std::string& filename,
        const cv::Mat& image)
    {
        if (directory.empty() || image.empty())
            return;

        try
        {
            const std::vector<int> params =
            {
                cv::IMWRITE_PNG_COMPRESSION, 1
            };

            cv::imwrite(
                (std::filesystem::path(directory) / filename).string(),
                image,
                params
            );
        }
        catch (...)
        {
            // Diagnostic saving must not interrupt processing.
        }
    }

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

    // Gray without copying when the input is already single-channel.
    bool toGrayView(const cv::Mat& input, cv::Mat& gray)
    {
        if (input.empty() || input.depth() != CV_8U)
            return false;

        if (input.channels() == 1)
            gray = input;
        else if (input.channels() == 3)
            cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
        else if (input.channels() == 4)
            cv::cvtColor(input, gray, cv::COLOR_BGRA2GRAY);
        else
            return false;

        return !gray.empty();
    }

    bool toBgrCopy(const cv::Mat& input, cv::Mat& bgr)
    {
        if (input.empty() || input.depth() != CV_8U)
            return false;

        if (input.channels() == 1)
            cv::cvtColor(input, bgr, cv::COLOR_GRAY2BGR);
        else if (input.channels() == 3)
            bgr = input.clone();
        else if (input.channels() == 4)
            cv::cvtColor(input, bgr, cv::COLOR_BGRA2BGR);
        else
            return false;

        return !bgr.empty();
    }

    std::vector<cv::Mat> makeKernels(const std::vector<int>& sizes)
    {
        std::vector<cv::Mat> kernels;

        for (int k : sizes)
        {
            if (k < 1)
                continue;

            kernels.push_back(
                cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(k, k)
                )
            );
        }

        if (kernels.empty())
        {
            kernels.push_back(
                cv::getStructuringElement(
                    cv::MORPH_ELLIPSE,
                    cv::Size(3, 3)
                )
            );
        }

        return kernels;
    }

    int makeBlockSize(const cv::Mat& image)
    {
        int blockSize =
            std::min(31, std::min(image.rows, image.cols));

        if (blockSize % 2 == 0)
            blockSize--;

        return std::max(3, blockSize);
    }

    // ============================================================
    // PARALLEL TASK INFRASTRUCTURE
    // One task = one (mask, kernel) pair. Results are reduced in the
    // original order so ties resolve exactly like the sequential code.
    // ============================================================

    struct MaskTask
    {
        const cv::Mat* mask = nullptr;
        const cv::Mat* kernel = nullptr;
    };

    struct TaskBest
    {
        double score = -1.0;
        std::vector<cv::Point> contour;
        int count = 0;
    };

    std::vector<MaskTask> buildTasks(
        const std::vector<const cv::Mat*>& masks,
        const std::vector<cv::Mat>& kernels,
        cv::Size requiredSize)
    {
        std::vector<MaskTask> tasks;

        for (const cv::Mat* mask : masks)
        {
            if (mask == nullptr || mask->empty())
                continue;

            if (mask->size() != requiredSize)
                continue;

            for (const cv::Mat& kernel : kernels)
            {
                MaskTask t;
                t.mask = mask;
                t.kernel = &kernel;
                tasks.push_back(t);
            }
        }

        return tasks;
    }

    void runTasks(
        int taskCount,
        bool parallel,
        const std::function<void(const cv::Range&)>& body)
    {
        if (taskCount <= 0)
            return;

        if (parallel && taskCount > 1)
            cv::parallel_for_(cv::Range(0, taskCount), body, taskCount);
        else
            body(cv::Range(0, taskCount));
    }

    bool reduceBest(
        const std::vector<TaskBest>& results,
        std::vector<cv::Point>& bestContour,
        int& totalContours)
    {
        totalContours = 0;

        double bestScore = -1.0;

        for (const TaskBest& tb : results)
        {
            totalContours += tb.count;

            if (!tb.contour.empty() && tb.score > bestScore)
            {
                bestScore = tb.score;
                bestContour = tb.contour;
            }
        }

        return !bestContour.empty();
    }

    // ============================================================
    // COMPONENT CANDIDATE (OpenCV-only method)
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

    bool evaluateComponent(
        const std::vector<cv::Point>& contour,
        cv::Size imageSize,
        double minSide,
        ComponentCandidate& candidate)
    {
        if (contour.size() < 5)
            return false;

        const double imageArea =
            static_cast<double>(imageSize.width) * imageSize.height;

        const double area = cv::contourArea(contour);

        if (area < imageArea * 0.00005)
            return false;

        if (area > imageArea * 0.60)
            return false;

        const cv::RotatedRect rotatedBox = cv::minAreaRect(contour);

        const double width = rotatedBox.size.width;
        const double height = rotatedBox.size.height;

        if (width < minSide || height < minSide)
            return false;

        const double longerSide = std::max(width, height);
        const double shorterSide = std::min(width, height);

        if (shorterSide <= 0.0)
            return false;

        const double aspectRatio = longerSide / shorterSide;

        if (aspectRatio > 4.5)
            return false;

        const cv::Rect box = cv::boundingRect(contour);

        if (box.width < minSide || box.height < minSide)
            return false;

        const double boxArea =
            static_cast<double>(box.width) * box.height;

        if (boxArea <= 0.0)
            return false;

        const double extent = area / boxArea;

        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);

        const double hullArea = cv::contourArea(hull);

        if (hullArea <= 0.0)
            return false;

        const double solidity = area / hullArea;

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
    // FIND MAIN COMPONENT (OpenCV-only method) - fast/parallel
    // ============================================================

    bool findMainComponentFast(
        const std::vector<const cv::Mat*>& masks,
        const std::vector<cv::Mat>& kernels,
        cv::Size imageSize,
        double minSide,
        bool parallel,
        std::vector<cv::Point>& bestContour,
        int& totalContours)
    {
        const std::vector<MaskTask> tasks =
            buildTasks(masks, kernels, imageSize);

        std::vector<TaskBest> results(tasks.size());

        const auto body = [&](const cv::Range& range)
            {
                cv::Mat processedMask;

                for (int t = range.start; t < range.end; ++t)
                {
                    TaskBest& tb = results[t];

                    cv::morphologyEx(
                        *tasks[t].mask,
                        processedMask,
                        cv::MORPH_CLOSE,
                        *tasks[t].kernel
                    );

                    std::vector<std::vector<cv::Point>> contours;

                    cv::findContours(
                        processedMask,
                        contours,
                        cv::RETR_EXTERNAL,
                        cv::CHAIN_APPROX_SIMPLE
                    );

                    tb.count = static_cast<int>(contours.size());

                    for (const auto& contour : contours)
                    {
                        ComponentCandidate candidate;

                        if (!evaluateComponent(
                            contour,
                            imageSize,
                            minSide,
                            candidate))
                        {
                            continue;
                        }

                        if (candidate.score > tb.score)
                        {
                            tb.score = candidate.score;
                            tb.contour = candidate.contour;
                        }
                    }
                }
            };

        runTasks(static_cast<int>(tasks.size()), parallel, body);

        return reduceBest(results, bestContour, totalContours);
    }

    // ============================================================
    // FIND BEST YOLO-GUIDED CONTOUR - fast/parallel
    // (same scoring as the original findBestYOLOGuidedContour)
    // ============================================================

    bool findBestYOLOGuidedContourFast(
        const std::vector<const cv::Mat*>& masks,
        const std::vector<cv::Mat>& kernels,
        const cv::Mat& yoloMask,
        const cv::Rect& yoloBounds,
        double yoloArea,
        const cv::RotatedRect& localYOLOBox,
        double minSide,
        bool parallel,
        std::vector<cv::Point>& bestContour,
        int& totalContours)
    {
        totalContours = 0;

        if (yoloArea <= 0.0)
            return false;

        const double referenceDiagonal =
            std::sqrt(
                static_cast<double>(localYOLOBox.size.width) * localYOLOBox.size.width +
                static_cast<double>(localYOLOBox.size.height) * localYOLOBox.size.height
            );

        if (referenceDiagonal <= 0.0)
            return false;

        const cv::Rect imageRect(0, 0, yoloMask.cols, yoloMask.rows);

        const std::vector<MaskTask> tasks =
            buildTasks(masks, kernels, yoloMask.size());

        std::vector<TaskBest> results(tasks.size());

        const auto body = [&](const cv::Range& range)
            {
                cv::Mat processedMask;
                cv::Mat patch;
                cv::Mat intersectionPatch;

                for (int t = range.start; t < range.end; ++t)
                {
                    TaskBest& tb = results[t];

                    cv::morphologyEx(
                        *tasks[t].mask,
                        processedMask,
                        cv::MORPH_CLOSE,
                        *tasks[t].kernel
                    );

                    std::vector<std::vector<cv::Point>> contours;

                    cv::findContours(
                        processedMask,
                        contours,
                        cv::RETR_EXTERNAL,
                        cv::CHAIN_APPROX_SIMPLE
                    );

                    tb.count = static_cast<int>(contours.size());

                    for (int idx = 0; idx < static_cast<int>(contours.size()); ++idx)
                    {
                        const std::vector<cv::Point>& contour = contours[idx];

                        if (contour.size() < 5)
                            continue;

                        const double contourArea = cv::contourArea(contour);

                        if (contourArea < yoloArea * 0.05)
                            continue;

                        if (contourArea > yoloArea * 2.50)
                            continue;

                        const cv::RotatedRect rotatedBox = cv::minAreaRect(contour);

                        const double width = rotatedBox.size.width;
                        const double height = rotatedBox.size.height;

                        if (width < minSide || height < minSide)
                            continue;

                        const double longerSide = std::max(width, height);
                        const double shorterSide = std::min(width, height);

                        if (shorterSide <= 0.0)
                            continue;

                        if (longerSide / shorterSide > 10.0)
                            continue;

                        // Cheap reject: must touch the YOLO region,
                        // otherwise overlapRatio would be 0 (< 0.30).
                        const cv::Rect br = cv::boundingRect(contour) & imageRect;

                        if (br.width <= 0 || br.height <= 0)
                            continue;

                        if ((br & yoloBounds).area() <= 0)
                            continue;

                        // Candidate mask only inside its bounding rect.
                        patch.create(br.size(), CV_8UC1);
                        patch.setTo(cv::Scalar(0));

                        cv::drawContours(
                            patch,
                            contours,
                            idx,
                            cv::Scalar(255),
                            cv::FILLED,
                            cv::LINE_8,
                            cv::noArray(),
                            INT_MAX,
                            cv::Point(-br.x, -br.y)
                        );

                        const double candidatePixelArea =
                            static_cast<double>(cv::countNonZero(patch));

                        if (candidatePixelArea <= 0.0)
                            continue;

                        cv::bitwise_and(
                            patch,
                            yoloMask(br),
                            intersectionPatch
                        );

                        const double intersectionArea =
                            static_cast<double>(cv::countNonZero(intersectionPatch));

                        const double unionArea =
                            candidatePixelArea + yoloArea - intersectionArea;

                        if (unionArea <= 0.0)
                            continue;

                        const double overlapRatio =
                            intersectionArea / candidatePixelArea;

                        const double yoloCoverage =
                            intersectionArea / yoloArea;

                        const double iou =
                            intersectionArea / unionArea;

                        if (overlapRatio < 0.30)
                            continue;

                        if (yoloCoverage < 0.20)
                            continue;

                        const cv::Moments moments = cv::moments(contour);

                        if (std::abs(moments.m00) < 1e-6)
                            continue;

                        const cv::Point2f candidateCentre(
                            static_cast<float>(moments.m10 / moments.m00),
                            static_cast<float>(moments.m01 / moments.m00)
                        );

                        const double normalizedDistance =
                            cv::norm(candidateCentre - localYOLOBox.center) /
                            referenceDiagonal;

                        const double centreScore =
                            std::exp(-2.0 * normalizedDistance);

                        const double areaRatio =
                            candidatePixelArea / yoloArea;

                        const double areaScore =
                            std::exp(
                                -0.65 *
                                std::abs(std::log(std::max(areaRatio, 1e-6)))
                            );

                        std::vector<cv::Point> hull;
                        cv::convexHull(contour, hull);

                        const double hullArea = cv::contourArea(hull);

                        if (hullArea <= 0.0)
                            continue;

                        const double solidity = contourArea / hullArea;

                        if (solidity < 0.30)
                            continue;

                        const double finalScore =
                            (0.25 * overlapRatio) +
                            (0.25 * yoloCoverage) +
                            (0.20 * iou) +
                            (0.15 * centreScore) +
                            (0.10 * areaScore) +
                            (0.05 * solidity);

                        if (finalScore > tb.score)
                        {
                            tb.score = finalScore;
                            tb.contour = contour;
                        }
                    }
                }
            };

        runTasks(static_cast<int>(tasks.size()), parallel, body);

        return reduceBest(results, bestContour, totalContours);
    }

    // ============================================================
    // SHARED PREPROCESSING: gray -> CLAHE -> blur -> masks
    // ============================================================

    struct MaskSet
    {
        cv::Mat enhanced;
        cv::Mat otsuNormal;
        cv::Mat otsuInverse;
        cv::Mat adaptiveNormal;
        cv::Mat adaptiveInverse;
        cv::Mat adaptiveInverseC2;
    };

    void buildMasks(
        const cv::Mat& workGray,
        bool needAllMasks,
        MaskSet& m)
    {
        getClahe()->apply(workGray, m.enhanced);

        cv::GaussianBlur(
            m.enhanced,
            m.enhanced,
            cv::Size(3, 3),
            0.8
        );

        const int blockSize = makeBlockSize(m.enhanced);

        cv::threshold(
            m.enhanced,
            m.otsuInverse,
            0,
            255,
            cv::THRESH_BINARY_INV | cv::THRESH_OTSU
        );

        cv::adaptiveThreshold(
            m.enhanced,
            m.adaptiveInverseC2,
            255,
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY_INV,
            blockSize,
            2
        );

        if (needAllMasks)
        {
            cv::threshold(
                m.enhanced,
                m.otsuNormal,
                0,
                255,
                cv::THRESH_BINARY | cv::THRESH_OTSU
            );

            cv::adaptiveThreshold(
                m.enhanced,
                m.adaptiveNormal,
                255,
                cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                cv::THRESH_BINARY,
                blockSize,
                4
            );

            cv::adaptiveThreshold(
                m.enhanced,
                m.adaptiveInverse,
                255,
                cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                cv::THRESH_BINARY_INV,
                blockSize,
                4
            );
        }
    }

    std::vector<const cv::Mat*> maskPointers(const MaskSet& m)
    {
        // Same order as the original code.
        return
        {
            &m.otsuNormal,
            &m.otsuInverse,
            &m.adaptiveNormal,
            &m.adaptiveInverse,
            &m.adaptiveInverseC2
        };
    }
}

// ============================================================
// OPENCV-ONLY COMPONENT SEGMENTATION (FAST)
// ============================================================

OpenCVOnlySegmentationResult
OpenCVOnlyGeometry::segmentFullImage(
    const cv::Mat& bgrImage,
    const SegmentationOptions& opt)
{
    using Clock = std::chrono::steady_clock;
    const auto tStart = Clock::now();

    OpenCVOnlySegmentationResult result;

    auto finish = [&]() -> OpenCVOnlySegmentationResult
        {
            result.processingTimeMs =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - tStart
                ).count();

            return result;
        };

    try
    {
        // ----------------------------------------------------
        // 1. VALIDATE
        // ----------------------------------------------------

        if (bgrImage.empty())
        {
            result.message = "Input image is empty.";
            return finish();
        }

        if (bgrImage.cols < 32 || bgrImage.rows < 32)
        {
            result.message = "Input image is too small.";
            return finish();
        }

        cv::Mat gray;

        if (!toGrayView(bgrImage, gray))
        {
            result.message = "Unsupported image format.";
            return finish();
        }

        // ----------------------------------------------------
        // 2. OPTIONAL DOWNSCALE
        // ----------------------------------------------------

        const int longSide = std::max(gray.cols, gray.rows);

        const double scale =
            (opt.maxImageSide > 0 && longSide > opt.maxImageSide)
            ? static_cast<double>(opt.maxImageSide) / longSide
            : 1.0;

        cv::Mat workGray;

        if (scale < 1.0)
            cv::resize(gray, workGray, cv::Size(), scale, scale, cv::INTER_AREA);
        else
            workGray = gray;

        // ----------------------------------------------------
        // 3. CLAHE + BLUR + MASKS
        // ----------------------------------------------------

        MaskSet m;

        buildMasks(workGray, opt.needAllMasks, m);

        result.grayImage = workGray;
        result.enhancedImage = m.enhanced;
        result.otsuNormalMask = m.otsuNormal;
        result.otsuInverseMask = m.otsuInverse;
        result.adaptiveNormalMask = m.adaptiveNormal;
        result.adaptiveInverseMask = m.adaptiveInverse;
        result.adaptiveInverseC2Mask = m.adaptiveInverseC2;

        // Combined masks are diagnostics only (not used for selection).
        if (opt.buildVisualizations || opt.saveDiagnostics)
        {
            cv::bitwise_or(
                m.otsuInverse,
                m.adaptiveInverseC2,
                result.combinedInverseMask
            );

            cv::morphologyEx(
                result.combinedInverseMask,
                result.closedCombinedInverseMask,
                cv::MORPH_CLOSE,
                cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3))
            );
        }

        // ----------------------------------------------------
        // 4. FIND MAIN COMPONENT (parallel)
        // ----------------------------------------------------

        const std::vector<cv::Mat> kernels = makeKernels(opt.kernelSizes);

        const double minSide =
            (scale < 1.0) ? std::max(2.0, 5.0 * scale) : 5.0;

        std::vector<cv::Point> contourWork;

        result.outerContourFound =
            findMainComponentFast(
                maskPointers(m),
                kernels,
                workGray.size(),
                minSide,
                opt.parallel,
                contourWork,
                result.contourCount
            );

        // ----------------------------------------------------
        // 5. MAP RESULT TO ORIGINAL IMAGE COORDINATES
        // ----------------------------------------------------

        if (result.outerContourFound)
        {
            if (scale < 1.0)
            {
                const double inv = 1.0 / scale;

                result.outerCableContour.reserve(contourWork.size());

                for (const cv::Point& p : contourWork)
                {
                    result.outerCableContour.emplace_back(
                        std::min(bgrImage.cols - 1, std::max(0, cvRound(p.x * inv))),
                        std::min(bgrImage.rows - 1, std::max(0, cvRound(p.y * inv)))
                    );
                }
            }
            else
            {
                result.outerCableContour = contourWork;
            }

            result.outerBoundingBox =
                cv::boundingRect(result.outerCableContour);

            result.outerRotatedBoundingBox =
                cv::minAreaRect(result.outerCableContour);

            result.outerCableMask =
                cv::Mat::zeros(bgrImage.size(), CV_8UC1);

            const std::vector<std::vector<cv::Point>> selected =
            {
                result.outerCableContour
            };

            cv::drawContours(
                result.outerCableMask,
                selected,
                0,
                cv::Scalar(255),
                cv::FILLED
            );

            result.message =
                "Main component candidate detected successfully.";
        }
        else
        {
            result.message =
                "No suitable component body was detected.";
        }

        // ----------------------------------------------------
        // 6. OPTIONAL VISUALIZATION / DISK OUTPUT (slow)
        // ----------------------------------------------------

        if (opt.buildVisualizations || opt.saveDiagnostics)
        {
            cv::Mat vis;

            if (toBgrCopy(bgrImage, vis))
            {
                result.candidateVisualization = vis;
                result.allContoursImage = vis.clone();

                if (result.outerContourFound)
                {
                    const std::vector<std::vector<cv::Point>> contours =
                    {
                        result.outerCableContour
                    };

                    cv::drawContours(
                        result.candidateVisualization,
                        contours,
                        0,
                        cv::Scalar(0, 0, 255),
                        2,
                        cv::LINE_AA
                    );

                    drawRotatedBox(
                        result.candidateVisualization,
                        result.outerRotatedBoundingBox
                    );

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
                }
            }

            if (opt.saveDiagnostics && !opt.outputDirectory.empty())
            {
                std::error_code ec;
                std::filesystem::create_directories(opt.outputDirectory, ec);

                saveFast(opt.outputDirectory, "01_grayscale.png", result.grayImage);
                saveFast(opt.outputDirectory, "02_enhanced.png", result.enhancedImage);
                saveFast(opt.outputDirectory, "03_otsu_normal.png", result.otsuNormalMask);
                saveFast(opt.outputDirectory, "04_otsu_inverse.png", result.otsuInverseMask);
                saveFast(opt.outputDirectory, "05_adaptive_normal.png", result.adaptiveNormalMask);
                saveFast(opt.outputDirectory, "06_adaptive_inverse.png", result.adaptiveInverseMask);
                saveFast(opt.outputDirectory, "07_adaptive_inverse_c2.png", result.adaptiveInverseC2Mask);
                saveFast(opt.outputDirectory, "08_combined_inverse.png", result.combinedInverseMask);
                saveFast(opt.outputDirectory, "09_combined_closed.png", result.closedCombinedInverseMask);

                if (result.outerContourFound)
                    saveFast(opt.outputDirectory, "10_component_mask.png", result.outerCableMask);

                saveFast(opt.outputDirectory, "11_component_result.png", result.candidateVisualization);
                saveFast(opt.outputDirectory, "12_component_contour.png", result.allContoursImage);

                if (result.outerContourFound)
                {
                    std::ofstream report(
                        (std::filesystem::path(opt.outputDirectory) /
                            "13_component_information.txt").string()
                    );

                    if (report.is_open())
                    {
                        report << "OPENCV COMPONENT DETECTION\n\n";
                        report << "Image width: " << bgrImage.cols << '\n';
                        report << "Image height: " << bgrImage.rows << '\n';
                        report << "Contours examined: " << result.contourCount << '\n';
                        report << "Contour points: " << result.outerCableContour.size() << '\n';
                        report << "Bounding box X: " << result.outerBoundingBox.x << '\n';
                        report << "Bounding box Y: " << result.outerBoundingBox.y << '\n';
                        report << "Bounding box width: " << result.outerBoundingBox.width << '\n';
                        report << "Bounding box height: " << result.outerBoundingBox.height << '\n';
                        report << "Centre X: " << result.outerRotatedBoundingBox.center.x << '\n';
                        report << "Centre Y: " << result.outerRotatedBoundingBox.center.y << '\n';
                        report << "Rotation angle: " << result.outerRotatedBoundingBox.angle << '\n';
                        report << "Rotated width: " << result.outerRotatedBoundingBox.size.width << '\n';
                        report << "Rotated height: " << result.outerRotatedBoundingBox.size.height << '\n';
                    }
                }
            }
        }

        result.success = true;

        return finish();
    }
    catch (const cv::Exception& e)
    {
        result.success = false;
        result.outerContourFound = false;
        result.message = std::string("OpenCV exception: ") + e.what();
        return finish();
    }
    catch (const std::exception& e)
    {
        result.success = false;
        result.outerContourFound = false;
        result.message = std::string("Processing exception: ") + e.what();
        return finish();
    }
    catch (...)
    {
        result.success = false;
        result.outerContourFound = false;
        result.message = "Unknown error during OpenCV segmentation.";
        return finish();
    }
}

// ============================================================
// OLD SIGNATURE: segmentFullImage(image, outputDirectory)
// Same behaviour as before (visualizations ON), but using the
// fast implementation.
// ============================================================

OpenCVOnlySegmentationResult
OpenCVOnlyGeometry::segmentFullImage(
    const cv::Mat& bgrImage,
    const std::string& outputDirectory)
{
    SegmentationOptions opt;

    opt.buildVisualizations = true;
    opt.saveDiagnostics = !outputDirectory.empty();
    opt.outputDirectory = outputDirectory;

    return segmentFullImage(bgrImage, opt);
}

// ============================================================
// YOLO + GEOMETRIC REFINEMENT (FAST HYBRID)
// ============================================================

YOLOGuidedRefinementResult
OpenCVOnlyGeometry::refineWithinYOLORegion(
    const cv::Mat& bgrImage,
    const cv::RotatedRect& yoloBoundingBox,
    const RefinementOptions& opt)
{
    using Clock = std::chrono::steady_clock;
    const auto tStart = Clock::now();

    YOLOGuidedRefinementResult result;

    auto finish = [&]() -> YOLOGuidedRefinementResult
        {
            result.processingTimeMs =
                std::chrono::duration<double, std::milli>(
                    Clock::now() - tStart
                ).count();

            return result;
        };

    try
    {
        // ----------------------------------------------------
        // 1. VALIDATE IMAGE
        // ----------------------------------------------------

        if (bgrImage.empty())
        {
            result.message = "Input image is empty.";
            return finish();
        }

        if (bgrImage.depth() != CV_8U)
        {
            result.message = "Image must have 8-bit pixel depth.";
            return finish();
        }

        if (bgrImage.cols < 32 || bgrImage.rows < 32)
        {
            result.message = "Input image is too small.";
            return finish();
        }

        const int ch = bgrImage.channels();

        if (ch != 1 && ch != 3 && ch != 4)
        {
            result.message = "Unsupported image format.";
            return finish();
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
            return finish();
        }

        // ----------------------------------------------------
        // 3. YOLO REGION -> EXPANDED ROI (clipped)
        // ----------------------------------------------------

        cv::Point2f yoloCorners[4];
        yoloBoundingBox.points(yoloCorners);

        const std::vector<cv::Point2f> yoloCornerVector(
            yoloCorners,
            yoloCorners + 4
        );

        const cv::Rect yoloRectangle =
            cv::boundingRect(yoloCornerVector);

        if (yoloRectangle.width <= 0 || yoloRectangle.height <= 0)
        {
            result.message = "Could not calculate YOLO region.";
            return finish();
        }

        const float paddingRatio =
            std::max(0.0f, std::min(0.75f, opt.paddingRatio));

        const int paddingX =
            std::max(12, static_cast<int>(std::ceil(yoloRectangle.width * paddingRatio)));

        const int paddingY =
            std::max(12, static_cast<int>(std::ceil(yoloRectangle.height * paddingRatio)));

        const cv::Rect expandedROI(
            yoloRectangle.x - paddingX,
            yoloRectangle.y - paddingY,
            yoloRectangle.width + 2 * paddingX,
            yoloRectangle.height + 2 * paddingY
        );

        const cv::Rect imageBounds(0, 0, bgrImage.cols, bgrImage.rows);

        result.roi = expandedROI & imageBounds;

        if (result.roi.width < 32 || result.roi.height < 32)
        {
            result.message = "YOLO ROI is too small.";
            return finish();
        }

        // ----------------------------------------------------
        // 4. ROI -> GRAY (no full-image clone)
        // ----------------------------------------------------

        cv::Mat roiGray;

        if (!toGrayView(bgrImage(result.roi), roiGray))
        {
            result.message = "Failed to extract YOLO region.";
            return finish();
        }

        // ----------------------------------------------------
        // 5. OPTIONAL DOWNSCALE
        // ----------------------------------------------------

        const int longSide = std::max(result.roi.width, result.roi.height);

        const double scale =
            (opt.maxRoiSide > 0 && longSide > opt.maxRoiSide)
            ? static_cast<double>(opt.maxRoiSide) / longSide
            : 1.0;

        cv::Mat workGray;

        if (scale < 1.0)
            cv::resize(roiGray, workGray, cv::Size(), scale, scale, cv::INTER_AREA);
        else
            workGray = roiGray;

        // ----------------------------------------------------
        // 6. CLAHE + BLUR + MASKS (no segmentFullImage call)
        // ----------------------------------------------------

        MaskSet m;

        buildMasks(workGray, opt.needAllMasks, m);

        // ----------------------------------------------------
        // 7. YOLO BOX -> WORKING (ROI-LOCAL, SCALED) COORDINATES
        // ----------------------------------------------------

        cv::RotatedRect localYOLOBox = yoloBoundingBox;

        localYOLOBox.center.x -= static_cast<float>(result.roi.x);
        localYOLOBox.center.y -= static_cast<float>(result.roi.y);

        if (scale < 1.0)
        {
            localYOLOBox.center.x *= static_cast<float>(scale);
            localYOLOBox.center.y *= static_cast<float>(scale);
            localYOLOBox.size.width *= static_cast<float>(scale);
            localYOLOBox.size.height *= static_cast<float>(scale);
        }

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
            cv::Mat::zeros(m.enhanced.size(), CV_8UC1);

        cv::fillPoly(
            yoloMask,
            std::vector<std::vector<cv::Point>>{ localPolygon },
            cv::Scalar(255)
        );

        const double yoloArea =
            static_cast<double>(cv::countNonZero(yoloMask));

        const cv::Rect yoloBounds =
            cv::boundingRect(localPolygon) &
            cv::Rect(0, 0, yoloMask.cols, yoloMask.rows);

        // ----------------------------------------------------
        // 8. FIND BEST YOLO-GUIDED CONTOUR (parallel)
        // ----------------------------------------------------

        const std::vector<cv::Mat> kernels = makeKernels(opt.kernelSizes);

        const double minSide =
            (scale < 1.0) ? std::max(2.0, 5.0 * scale) : 5.0;

        std::vector<cv::Point> selectedContour;
        int totalContours = 0;

        const bool found =
            findBestYOLOGuidedContourFast(
                maskPointers(m),
                kernels,
                yoloMask,
                yoloBounds,
                yoloArea,
                localYOLOBox,
                minSide,
                opt.parallel,
                selectedContour,
                totalContours
            );

        result.segmentation.contourCount = totalContours;

        if (!found)
        {
            result.message =
                "YOLO detected the component, but OpenCV could "
                "not find a suitable matching contour. "
                "Inspect the ROI threshold images.";

            return finish();
        }

        // ----------------------------------------------------
        // 9. REFINE CONTOUR (same approximation as the original)
        // ----------------------------------------------------

        std::vector<cv::Point> refinedContourWork;

        const double perimeter =
            cv::arcLength(selectedContour, true);

        const double epsilon =
            std::max(0.5, perimeter * 0.0005);

        cv::approxPolyDP(
            selectedContour,
            refinedContourWork,
            epsilon,
            true
        );

        if (refinedContourWork.size() < 5)
            refinedContourWork = selectedContour;

        // Map back to ROI-local coordinates at ORIGINAL scale.
        std::vector<cv::Point> refinedContour;
        refinedContour.reserve(refinedContourWork.size());

        if (scale < 1.0)
        {
            const double inv = 1.0 / scale;

            for (const cv::Point& p : refinedContourWork)
            {
                refinedContour.emplace_back(
                    cvRound(p.x * inv),
                    cvRound(p.y * inv)
                );
            }
        }
        else
        {
            refinedContour = refinedContourWork;
        }

        const cv::RotatedRect refinedLocalBox =
            cv::minAreaRect(refinedContour);

        const cv::Rect refinedLocalBoundingBox =
            cv::boundingRect(refinedContour);

        // ----------------------------------------------------
        // 10. ROI-LOCAL SEGMENTATION INFO (cheap)
        // ----------------------------------------------------

        result.segmentation.success = true;
        result.segmentation.outerContourFound = true;
        result.segmentation.outerCableContour = refinedContour;
        result.segmentation.outerBoundingBox = refinedLocalBoundingBox;
        result.segmentation.outerRotatedBoundingBox = refinedLocalBox;
        result.segmentation.message = "Fast refinement completed.";

        // ----------------------------------------------------
        // 11. FULL IMAGE COORDINATES
        // ----------------------------------------------------

        result.refinedContour = refinedContour;

        for (cv::Point& point : result.refinedContour)
        {
            point.x += result.roi.x;
            point.y += result.roi.y;
        }

        result.refinedBoundingBox = refinedLocalBoundingBox;
        result.refinedBoundingBox.x += result.roi.x;
        result.refinedBoundingBox.y += result.roi.y;

        result.refinedRotatedBoundingBox = refinedLocalBox;
        result.refinedRotatedBoundingBox.center.x += static_cast<float>(result.roi.x);
        result.refinedRotatedBoundingBox.center.y += static_cast<float>(result.roi.y);

        result.refinedContourFound = true;
        result.success = true;
        result.message = "YOLO detection and geometric refinement completed.";

        // ----------------------------------------------------
        // 12. OPTIONAL VISUALIZATIONS / DISK OUTPUT (slow)
        // ----------------------------------------------------

        if (opt.buildVisualizations || opt.saveDiagnostics)
        {
            cv::Mat displaySource;

            if (toBgrCopy(bgrImage, displaySource))
            {
                const cv::Mat roiImage = displaySource(result.roi).clone();

                cv::RotatedRect drawYOLOBox = yoloBoundingBox;
                drawYOLOBox.center.x -= static_cast<float>(result.roi.x);
                drawYOLOBox.center.y -= static_cast<float>(result.roi.y);

                const std::vector<std::vector<cv::Point>> refinedLocalContours =
                {
                    refinedContour
                };

                result.segmentation.outerCableMask =
                    cv::Mat::zeros(roiImage.size(), CV_8UC1);

                cv::drawContours(
                    result.segmentation.outerCableMask,
                    refinedLocalContours,
                    0,
                    cv::Scalar(255),
                    cv::FILLED
                );

                result.segmentation.grayImage = workGray;
                result.segmentation.enhancedImage = m.enhanced;
                result.segmentation.otsuNormalMask = m.otsuNormal;
                result.segmentation.otsuInverseMask = m.otsuInverse;
                result.segmentation.adaptiveNormalMask = m.adaptiveNormal;
                result.segmentation.adaptiveInverseMask = m.adaptiveInverse;
                result.segmentation.adaptiveInverseC2Mask = m.adaptiveInverseC2;

                result.roiVisualization = roiImage.clone();

                drawRotatedBox(result.roiVisualization, drawYOLOBox, cv::Scalar(0, 255, 255), 2);

                cv::drawContours(
                    result.roiVisualization,
                    refinedLocalContours,
                    0,
                    cv::Scalar(0, 0, 255),
                    2,
                    cv::LINE_AA
                );

                drawRotatedBox(result.roiVisualization, refinedLocalBox, cv::Scalar(255, 0, 0), 2);

                cv::circle(
                    result.roiVisualization,
                    refinedLocalBox.center,
                    5,
                    cv::Scalar(0, 255, 0),
                    cv::FILLED
                );

                result.fullImageVisualization = displaySource;

                drawRotatedBox(result.fullImageVisualization, yoloBoundingBox, cv::Scalar(0, 255, 255), 2);

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

                drawRotatedBox(
                    result.fullImageVisualization,
                    result.refinedRotatedBoundingBox,
                    cv::Scalar(255, 0, 0),
                    2
                );

                cv::circle(
                    result.fullImageVisualization,
                    result.refinedRotatedBoundingBox.center,
                    6,
                    cv::Scalar(0, 255, 0),
                    cv::FILLED
                );
            }

            if (opt.saveDiagnostics && !opt.outputDirectory.empty())
            {
                std::error_code ec;
                std::filesystem::create_directories(opt.outputDirectory, ec);

                saveFast(opt.outputDirectory, "01_grayscale.png", workGray);
                saveFast(opt.outputDirectory, "02_enhanced.png", m.enhanced);
                saveFast(opt.outputDirectory, "03_otsu_normal.png", m.otsuNormal);
                saveFast(opt.outputDirectory, "04_otsu_inverse.png", m.otsuInverse);
                saveFast(opt.outputDirectory, "05_adaptive_normal.png", m.adaptiveNormal);
                saveFast(opt.outputDirectory, "06_adaptive_inverse.png", m.adaptiveInverse);
                saveFast(opt.outputDirectory, "07_adaptive_inverse_c2.png", m.adaptiveInverseC2);
                saveFast(opt.outputDirectory, "14_yolo_guided_refinement_roi.png", result.roiVisualization);
                saveFast(opt.outputDirectory, "15_yolo_geometric_final_result.png", result.fullImageVisualization);

                std::ofstream report(
                    (std::filesystem::path(opt.outputDirectory) /
                        "16_yolo_geometric_information.txt").string()
                );

                if (report.is_open())
                {
                    report << "YOLO + GEOMETRIC REFINEMENT\n\n";
                    report << "YOLO centre X: " << yoloBoundingBox.center.x << '\n';
                    report << "YOLO centre Y: " << yoloBoundingBox.center.y << '\n';
                    report << "YOLO width: " << yoloBoundingBox.size.width << '\n';
                    report << "YOLO height: " << yoloBoundingBox.size.height << '\n';
                    report << "YOLO angle: " << yoloBoundingBox.angle << '\n';
                    report << "\nREFINED GEOMETRY\n\n";
                    report << "Refined contour points: " << result.refinedContour.size() << '\n';
                    report << "Refined centre X: " << result.refinedRotatedBoundingBox.center.x << '\n';
                    report << "Refined centre Y: " << result.refinedRotatedBoundingBox.center.y << '\n';
                    report << "Refined width: " << result.refinedRotatedBoundingBox.size.width << '\n';
                    report << "Refined height: " << result.refinedRotatedBoundingBox.size.height << '\n';
                    report << "Refined angle: " << result.refinedRotatedBoundingBox.angle << '\n';
                    report << "Contours examined: " << totalContours << '\n';
                }
            }
        }

        return finish();
    }
    catch (const cv::Exception& e)
    {
        result.success = false;
        result.refinedContourFound = false;
        result.message = std::string("OpenCV exception: ") + e.what();
        return finish();
    }
    catch (const std::exception& e)
    {
        result.success = false;
        result.refinedContourFound = false;
        result.message = std::string("Processing exception: ") + e.what();
        return finish();
    }
    catch (...)
    {
        result.success = false;
        result.refinedContourFound = false;
        result.message = "Unknown error during YOLO geometric refinement.";
        return finish();
    }
}

// ============================================================
// OLD SIGNATURE: refineWithinYOLORegion(image, box, dir, padding)
// ============================================================

YOLOGuidedRefinementResult
OpenCVOnlyGeometry::refineWithinYOLORegion(
    const cv::Mat& bgrImage,
    const cv::RotatedRect& yoloBoundingBox,
    const std::string& outputDirectory,
    float paddingRatio)
{
    RefinementOptions opt;

    opt.paddingRatio = paddingRatio;
    opt.saveDiagnostics = !outputDirectory.empty();
    opt.buildVisualizations = !outputDirectory.empty();
    opt.outputDirectory = outputDirectory;

    return refineWithinYOLORegion(bgrImage, yoloBoundingBox, opt);
}