
#include "YOLODetector.h"

#include <algorithm>
#include <cmath>
#include <iostream>

YOLODetector::YOLODetector(const std::wstring& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "XRayAutoCentering"),
    session(nullptr),
    loaded(false)
{
    try
    {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(
            GraphOptimizationLevel::ORT_ENABLE_ALL
        );

        session = new Ort::Session(
            env, modelPath.c_str(), options
        );

        if (session->GetInputCount() < 1 ||
            session->GetOutputCount() < 1)
        {
            throw std::runtime_error(
                "Model has no input or output tensors."
            );
        }

        auto inInfo = session->GetInputTypeInfo(0)
            .GetTensorTypeAndShapeInfo();
        inputShape = inInfo.GetShape();

        auto outInfo = session->GetOutputTypeInfo(0)
            .GetTensorTypeAndShapeInfo();
        outputShape = outInfo.GetShape();

        loaded = true;

        std::cout << "ONNX model loaded.\n";
        std::cout << "Input shape: ";
        for (auto d : inputShape) std::cout << d << ' ';
        std::cout << "\nOutput shape: ";
        for (auto d : outputShape) std::cout << d << ' ';
        std::cout << '\n';
    }
    catch (const std::exception& e)
    {
        std::cerr << "Model loading failed: "
            << e.what() << '\n';

        delete session;
        session = nullptr;
        loaded = false;
    }
}

YOLODetector::~YOLODetector()
{
    delete session;
}

bool YOLODetector::isLoaded() const
{
    return loaded;
}

size_t YOLODetector::getInputCount() const
{
    return loaded ? session->GetInputCount() : 0;
}

size_t YOLODetector::getOutputCount() const
{
    return loaded ? session->GetOutputCount() : 0;
}

std::vector<int64_t> YOLODetector::getInputShape() const
{
    return inputShape;
}

std::vector<int64_t> YOLODetector::getOutputShape() const
{
    return outputShape;
}

CableDetection YOLODetector::detect(const cv::Mat& image)
{
    CableDetection result;

    if (!loaded || image.empty())
        return result;

    try
    {
        constexpr int inputSize = 640;
        constexpr float confidenceThreshold = 0.25f;

        // Match Ultralytics-style letterboxing.
        const float scale = std::min(
            static_cast<float>(inputSize) / image.cols,
            static_cast<float>(inputSize) / image.rows
        );

        const int resizedWidth =
            static_cast<int>(std::round(image.cols * scale));
        const int resizedHeight =
            static_cast<int>(std::round(image.rows * scale));

        cv::Mat resized;
        cv::resize(
            image, resized,
            cv::Size(resizedWidth, resizedHeight)
        );

        const int padX = (inputSize - resizedWidth) / 2;
        const int padY = (inputSize - resizedHeight) / 2;

        cv::Mat padded(
            inputSize, inputSize, CV_8UC3,
            cv::Scalar(114, 114, 114)
        );

        resized.copyTo(
            padded(cv::Rect(
                padX, padY, resizedWidth, resizedHeight
            ))
        );

        // Convert BGR to RGB, normalize to [0,1],
        // and create NCHW tensor: 1x3x640x640.
        cv::Mat rgb;
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

        cv::Mat floatImage;
        rgb.convertTo(floatImage, CV_32F, 1.0 / 255.0);

        std::vector<float> inputTensor(
            3 * inputSize * inputSize
        );

        for (int y = 0; y < inputSize; ++y)
        {
            for (int x = 0; x < inputSize; ++x)
            {
                const cv::Vec3f pixel =
                    floatImage.at<cv::Vec3f>(y, x);

                const size_t index =
                    static_cast<size_t>(y) * inputSize + x;

                inputTensor[index] =
                    pixel[0];

                inputTensor[
                    static_cast<size_t>(inputSize) * inputSize
                        + index
                ] = pixel[1];

                inputTensor[
                    2ULL * inputSize * inputSize + index
                ] = pixel[2];
            }
        }

        const std::array<int64_t, 4> tensorShape = {
            1, 3, inputSize, inputSize
        };

        auto memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault
        );

        auto input = Ort::Value::CreateTensor<float>(
            memoryInfo,
            inputTensor.data(),
            inputTensor.size(),
            tensorShape.data(),
            tensorShape.size()
        );

        Ort::AllocatorWithDefaultOptions allocator;

        auto inputName =
            session->GetInputNameAllocated(0, allocator);
        auto outputName =
            session->GetOutputNameAllocated(0, allocator);

        const char* inputNames[] = { inputName.get() };
        const char* outputNames[] = { outputName.get() };

        auto outputs = session->Run(
            Ort::RunOptions{ nullptr },
            inputNames, &input, 1,
            outputNames, 1
        );

        if (outputs.empty() || !outputs[0].IsTensor())
            return result;

        auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto shape = info.GetShape();

        // Expected output: [1, 6, 8400].
        if (shape.size() != 3 ||
            shape[0] != 1 ||
            shape[1] != 6 ||
            shape[2] <= 0)
        {
            std::cerr << "Unexpected ONNX output shape.\n";
            return result;
        }

        const float* data =
            outputs[0].GetTensorData<float>();

        const int candidates = static_cast<int>(shape[2]);

        float bestConfidence = confidenceThreshold;
        int bestIndex = -1;

        // Output is channel-first: [cx, cy, w, h, score, angle].
        for (int i = 0; i < candidates; ++i)
        {
            const float confidence =
                data[4 * candidates + i];

            if (std::isfinite(confidence) &&
                confidence > bestConfidence)
            {
                bestConfidence = confidence;
                bestIndex = i;
            }
        }

        if (bestIndex < 0)
            return result;

        const float cx = data[bestIndex];
        const float cy = data[candidates + bestIndex];
        const float width = data[2 * candidates + bestIndex];
        const float height = data[3 * candidates + bestIndex];
        const float angle = data[5 * candidates + bestIndex];

        if (!std::isfinite(cx) || !std::isfinite(cy) ||
            !std::isfinite(width) || !std::isfinite(height) ||
            !std::isfinite(angle) || width <= 0 || height <= 0)
        {
            return result;
        }

        // Undo letterbox padding and scaling.
        result.center.x = (cx - padX) / scale;
        result.center.y = (cy - padY) / scale;
        result.size.width = width / scale;
        result.size.height = height / scale;
        result.angleRadians = angle;
        result.confidence = bestConfidence;
        result.found = true;
    }
    catch (const Ort::Exception& e)
    {
        std::cerr << "Inference error: "
            << e.what() << '\n';
    }
    catch (const std::exception& e)
    {
        std::cerr << "Detection error: "
            << e.what() << '\n';
    }

    return result;
}
