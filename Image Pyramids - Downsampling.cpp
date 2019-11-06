#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

cv::Mat downsample(cv::Mat img){
    cv::Mat img_d(img.rows/2, img.cols/2, CV_8U);
    for(int i=0; i<img_d.rows; i++){
        for(int j=0; j<img_d.cols; j++){
            img_d.at<uchar>(i,j) = img.at<uchar>(i*2, j*2);
        }
    }
    return img_d;
}

cv::Mat downsample_blur(cv::Mat img){
    cv::Mat image_gaussian;
    cv::GaussianBlur(img, image_gaussian, cv::Size(5,5), 0);
    cv::Mat img_d(image_gaussian.rows/2, image_gaussian.cols/2, CV_8U);
    for(int i=0; i<img_d.rows; i++){
        for(int j=0; j<img_d.cols; j++){
            img_d.at<uchar>(i,j) = image_gaussian.at<uchar>(i*2, j*2);
        }
    }
    return img_d;
}

int main()
{
    cv::Mat img = cv::imread("image_1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_d1, img_db1;
    cv::Mat img_d2, img_db2;
    cv::Mat img_d3, img_db3;

    img_d1 = downsample(img);
    img_db1 = downsample_blur(img);

    img_d2 = downsample(img_d1);
    img_db2 = downsample_blur(img_db1);

    img_d3 = downsample(img_d2);
    img_db3 = downsample_blur(img_db2);

    cv::resize(img_d1, img_d1, img.size());
    cv::resize(img_d2, img_d2, img.size());
    cv::resize(img_d3, img_d3, img.size());

    cv::resize(img_db1, img_db1, img.size());
    cv::resize(img_db2, img_db2, img.size());
    cv::resize(img_db3, img_db3, img.size());

    // Laplacian images
    cv::Mat img_difference = img - img_db1;
    cv::Mat img_difference1 = img_db1 - img_db2;
    cv::Mat img_difference2 = img_db2 - img_db3;

    // Reconstruction
    cv::Mat reconstruct2 = img_db3 + img_difference2;
    cv::Mat reconstruct1 = img_db2 + img_difference1;
    cv::Mat reconstruct = img_db1 + img_difference;

    return 0;
}
