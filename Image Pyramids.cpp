#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

int main()
{
    cv::Mat img = cv::imread("image_1.png");

    cv::pyrUp(img, img, cv::Size(img.cols*2, img.rows*2));

    cv::pyrDown(img, img, cv::Size(img.cols/2, img.rows/2));

    return 0;
}
