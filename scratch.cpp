#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

enum class SourceMode {
    camera,
    photo,
};

enum class FilterMode {
    kNone,
    kGray,
    kBinary,
    kKodak,
    kVision3,
};

struct Config {
    int camera_index = 1;
    FilterMode mode = FilterMode::kNone;
    int threshold = 128;
    float grain_strength = 6.0f;
};

std::string ModeToString(FilterMode mode) {
    switch(mode) {
        case FilterMode::kNone:    return "none";
        case FilterMode::kGray:    return "gray";
        case FilterMode::kBinary:  return "binary";
        case FilterMode::kKodak:   return "kodak";
        case FilterMode::kVision3: return "vision3";
    }
    return "unknown";
}

void DrawOverlay(cv::Mat& frame, const Config& config) {
    std::string line1 = "Mode: " + ModeToString(config.mode);
    std::string line2 = "g(gray) b(binary) k(kodak) v(vision3) n(normal) q(quit)";
    cv::putText(frame, line1, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255,255,255), 2);
    cv::putText(frame, line2, cv::Point(20, frame.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 1);
}

void ApplyKodakFilter(const cv::Mat& input, cv::Mat& output) {
    // color mixing kernel
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        1.05f, -0.05f, 0.00f,
        -0.02f, 1.03f, -0.01f,
        0.00f, -0.03f, 1.04f);
    cv::transform(input, output, kernel);
    cv::convertScaleAbs(output, output, 1.10, 6.0); //create a warmer look
}

void ApplyFilmCurve(cv::Mat& image) {
    cv::Mat lut(1, 256, CV_8UC1);
    for (int i = 0; i < 256; i++) {
        float x = i / 255.0f;
        float y = 1.0f / (1.0f + exp(-6.0f * (x - 0.5f)));
        y = pow(y, 0.95f);
        lut.at<uchar>(i) = static_cast<uchar>(y * 255);
    }

    cv::LUT(image, lut, image);
}

void ApplyFilmGrain(cv::Mat& image, float strength) {
    cv::Mat noise(image.size(), CV_32FC3);
    cv::randn(noise, 0, strength);
    cv::Mat float_img;
    image.convertTo(float_img, CV_32FC3);
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    gray.convertTo(gray, CV_32F, 1.0/255.0);
    cv::Mat mask = 1.0 - gray; // more grain in shadows
    cv::cvtColor(mask, mask, cv::COLOR_GRAY2BGR);
    float_img += noise.mul(mask);
    float_img.convertTo(image, CV_8UC3);
}

void ApplyHalation(cv::Mat& image) {
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv::Mat mask;
    cv::threshold(gray, mask, 245, 255, cv::THRESH_BINARY);
    cv::GaussianBlur(mask, mask, cv::Size(25,25), 0);
    cv::Mat mask3;
    cv::cvtColor(mask, mask3, cv::COLOR_GRAY2BGR);

    std::vector<cv::Mat> ch;
    cv::split(mask3, ch);
    ch[2] *= 0.6f; // red
    ch[1] *= 0.2f; // green
    ch[0] *= 0.1f; // blue

    cv::merge(ch, mask3);
    cv::addWeighted(image, 1.0, mask3, 0.05, 0, image);
}

void ApplyVignette(cv::Mat& image) {
    int cx = image.cols / 2;
    int cy = image.rows / 2;
    cv::Mat mask(image.size(), CV_32FC1);
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            float dx = (x - cx) / (float)cx;
            float dy = (y - cy) / (float)cy;
            float dist = std::sqrt(dx*dx + dy*dy);
            float v = 1.0f - std::min(1.0f, dist * 0.45f);
            mask.at<float>(y, x) = v;
        }
    }
    cv::Mat mask3ch;
    cv::cvtColor(mask, mask3ch, cv::COLOR_GRAY2BGR);
    cv::Mat float_image; 
    image.convertTo(float_image, CV_32FC3);
    float_image = float_image.mul(mask3ch);
    float_image.convertTo(image, CV_8UC3);
}

void ApplyVision3Filter(const cv::Mat& input, cv::Mat& output, const Config& config) {
    // color mixing kernel
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
        1.33f, -0.41f, 0.09f,
        -0.12f, 1.27f, -0.16f,
        -0.04f, -0.21f, 1.26f);
    cv::transform(input, output, kernel);
    ApplyFilmCurve(output);
    ApplyFilmGrain(output, config.grain_strength);
    ApplyHalation(output);
    ApplyVignette(output);
}

int main() {
    Config config;
    cv::VideoCapture cap(config.camera_index);

    if (!cap.isOpened()) {
        std::cerr << "Fail to open camera index " << config.camera_index << "\\n";
        return 1; //the program failed
    }
    cv::Mat frame; //openCV frame type (2D array of pixels)
    cv::Mat greyScaleMat;
    cv::Mat output;

    while(true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Failed to read frame from the camera.\\n";
            break;
        }
        cv::flip(frame, frame, 1); //flip so that it is not mirrored

        if (config.mode == FilterMode::kGray) {
            cv::cvtColor(frame, greyScaleMat, cv::COLOR_BGR2GRAY);
            cv::cvtColor(greyScaleMat, output, cv::COLOR_GRAY2BGR);
        } else if (config.mode == FilterMode::kNone) {
            frame.copyTo(output);
        } else if (config.mode == FilterMode::kBinary) {
            cv::cvtColor(frame, greyScaleMat, cv::COLOR_BGR2GRAY);
            cv::threshold(greyScaleMat, greyScaleMat, config.threshold, 255, cv::THRESH_BINARY);
            cv::cvtColor(greyScaleMat, output, cv::COLOR_GRAY2BGR);
        } else if (config.mode == FilterMode::kKodak) {
            ApplyKodakFilter(frame, output);
        } else if (config.mode == FilterMode::kVision3) {
            ApplyVision3Filter(frame, output, config);
        }

        DrawOverlay(output, config);
        cv::imshow("KODAK-Inspired Camera Filters", output);

        const int key = cv::waitKey(1);
        if (key == 'q' || key == 'Q') break;
        if (key == 'g' || key == 'G') config.mode = FilterMode::kGray;
        if (key == 'n' || key == 'N') config.mode = FilterMode::kNone;
        if (key == 'b' || key == 'B') config.mode = FilterMode::kBinary;
        if (key == '[') config.threshold = std::max(0, config.threshold - 5);
        if (key == ']') config.threshold = std::min(255, config.threshold + 5);
        if (key == 'k' || key == 'K') config.mode = FilterMode::kKodak;
        if (key == 'v' || key == 'V') config.mode = FilterMode::kVision3;

    cap.release();
    cv::destroyAllWindows();
    return 0;
}