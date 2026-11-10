
#include "GeometricRefinement.h"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <exception>

GeometricRefinementResult GeometricRefinement::processFullImage(
    const cv::Mat& originalImage,
    const std::string& outputDirectory)
{
    GeometricRefinementResult result;

    try
    {
        // G1.1: Validate input
        if (originalImage.empty())
        {
            result.message = "Input image is empty.";
            return result;
        }

        if (originalImage.depth() != CV_8U)
        {
            result.message = "Input image must use 8-bit pixel depth.";
            return result;
        }

        if (originalImage.channels() != 1 &&
            originalImage.channels() != 3 &&
            originalImage.channels() != 4)
        {
            result.message = "Unsupported image format.";
            return result;
        }

        // G1.2: Preserve original image
        result.originalImage = originalImage.clone();

        // G1.3: Convert to grayscale
        if (originalImage.channels() == 3)
        {
            cv::cvtColor(
                originalImage,
                result.grayscaleImage,
                cv::COLOR_BGR2GRAY
            );
        }
        else if (originalImage.channels() == 4)
        {
            cv::cvtColor(
                originalImage,
                result.grayscaleImage,
                cv::COLOR_BGRA2GRAY
            );
        }
        else
        {
            result.grayscaleImage = originalImage.clone();
        }

        // G1.4: CLAHE enhancement
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(
            2.0,
            cv::Size(8, 8)
        );

        clahe->apply(
            result.grayscaleImage,
            result.enhancedImage
        );

        // G1.5: Gaussian blur
        cv::GaussianBlur(
            result.enhancedImage,
            result.blurredImage,
            cv::Size(3, 3),
            0.8
        );

        // G1.6: Save diagnostic images
        if (!outputDirectory.empty())
        {
            std::filesystem::create_directories(outputDirectory);

            const std::string separator = "/";

            bool saved =
                cv::imwrite(
                    outputDirectory + separator + "g1_original.png",
                    result.originalImage
                ) &&
                cv::imwrite(
                    outputDirectory + separator + "g1_grayscale.png",
                    result.grayscaleImage
                ) &&
                cv::imwrite(
                    outputDirectory + separator + "g1_enhanced.png",
                    result.enhancedImage
                ) &&
                cv::imwrite(
                    outputDirectory + separator + "g1_blurred.png",
                    result.blurredImage
                );

            if (!saved)
            {
                result.message =
                    "Preprocessing completed, but diagnostic images could not all be saved.";

                return result;
            }
        }

        result.success = true;
        result.message = "G1 preprocessing completed successfully.";

        return result;
    }
    catch (const cv::Exception& e)
    {
        result.message = std::string("OpenCV error: ") + e.what();
        return result;
    }
    catch (const std::exception& e)
    {
        result.message = std::string("Processing error: ") + e.what();
        return result;
    }
}
