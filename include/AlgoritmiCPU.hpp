#pragma once
#include <opencv2/opencv.hpp>

void runSobelCPU(const cv::Mat& input, cv::Mat& output);
void runBlurCPU(const cv::Mat& input, cv::Mat& output);
void runErosionCPU(const cv::Mat& input, cv::Mat& output);
void runDilationCPU(const cv::Mat& input, cv::Mat& output);
void runTranslationCPU(const cv::Mat& input, cv::Mat& output, int dx, int dy);
void runRotationCPU(const cv::Mat& input, cv::Mat& output, float grado_rotazione);
void runScalingCPU(const cv::Mat& input, cv::Mat& output, float scala);
        