
#pragma once

#include <opencv2/opencv.hpp>
#include <string>

struct GeometricRefinementResult
{
    bool success = false;
    std::string message;

    cv::Mat originalImage;
    cv::Mat grayscaleImage;
    cv::Mat enhancedImage;
    cv::Mat blurredImage;
};

class GeometricRefinement
{
public:
    static GeometricRefinementResult processFullImage(
        const cv::Mat& originalImage,
        const std::string& outputDirectory = ""
    );
};
#pragma once
