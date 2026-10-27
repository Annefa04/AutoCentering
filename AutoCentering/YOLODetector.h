
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct CableDetection
{
    cv::Point2f center;
    cv::Size2f size;
    float angleRadians = 0.0f;
    float confidence = 0.0f;
    bool found = false;
};

class YOLODetector
{
public:
    explicit YOLODetector(const std::wstring& modelPath);
    ~YOLODetector();

    bool isLoaded() const;

    size_t getInputCount() const;
    size_t getOutputCount() const;

    std::vector<int64_t> getInputShape() const;
    std::vector<int64_t> getOutputShape() const;

    CableDetection detect(const cv::Mat& image);

private:
    Ort::Env env;
    Ort::Session* session;
    bool loaded;

    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
};
