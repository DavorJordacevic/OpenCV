#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/features2d.hpp>

using namespace std;

int main()
{
    // Loading the actual image
    cv::Mat image_left, image_right;

    image_left=imread("left.jpg", cv::IMREAD_COLOR);
    cv::imshow("Left image", image_left);
    cv::waitKey();

    image_right=imread("right.jpg", cv::IMREAD_COLOR);
    cv::imshow("Right image", image_right);
    cv::waitKey();

    // Converting the color image into grayscale
    cvtColor(image_left, image_left, cv::COLOR_BGR2GRAY);
    cvtColor(image_right,image_right,cv::COLOR_BGR2GRAY);

    // Vector of keypoints
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;

    // We will use orb feature detector
    // Construction of the orb feature detector
    cv::Ptr<cv::FeatureDetector> orb = cv::ORB::create(5000);

    // Feature detection
    orb->detect(image_left, keypoints1);
    orb->detect(image_right,keypoints2);

    // Draw the kepoints
    cv::Mat imageKeyPoints;
    // Left image
    cv::drawKeypoints(image_left, keypoints1, imageKeyPoints, cv::Scalar(0, 0, 255),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("Left image", imageKeyPoints);
    cv::waitKey();
    // Right image
    cv::drawKeypoints(image_right, keypoints2, imageKeyPoints, cv::Scalar(0, 0, 255),
                      cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow("Right image", imageKeyPoints);
    cv::waitKey();

    // Extraction of the orb descriptors
    cv::Mat descriptors1, descriptors2;
    orb->compute(image_left, keypoints1, descriptors1);
    orb->compute(image_right, keypoints2, descriptors2);

    cv::FlannBasedMatcher matcher = cv::FlannBasedMatcher(cv::makePtr<cv::flann::LshIndexParams>(6, 12, 1),cv::makePtr<cv::flann::SearchParams>(50));
    std::vector< std::vector<cv::DMatch> > matches;
    matcher.knnMatch(descriptors1, descriptors2, matches, 2 );

    std::cout<<"matches size: "<<matches.size()<<std::endl;
    // Select few Matches
    std::vector<cv::DMatch> good;

    //Filter matches using the Lowe's ratio test
    const float ratio_thresh = 0.7f;
    for (size_t i = 0; i < matches.size(); i++){
        if (matches[i].size() >= 2){
            if (matches[i][0].distance < ratio_thresh * matches[i][1].distance){
                good.push_back(matches[i][0]);
            }
        }
    }

    cout<< "Number of matched points: " << good.size() << endl;

    // Convert one vector of keypoints into two vectors of Point2f
    std::vector<int> points1;
    std::vector<int> points2;
    for (auto iter = good.begin(); iter != good.end(); ++iter){
        // Get the indexes of the selected matched keypoints
        points1.push_back(iter->queryIdx);
        points2.push_back(iter->trainIdx);
    }

    // Convert keypoints into Point2f
    std::vector<cv::Point2f> selPoints1, selPoints2;
    cv::KeyPoint::convert(keypoints1, selPoints1, points1);
    cv::KeyPoint::convert(keypoints2, selPoints2, points2);


    // Compute F matrix from all good matches
    cv::Mat fundemental = cv::findFundamentalMat(cv::Mat(selPoints1), cv::Mat(selPoints2), cv::FM_LMEDS);

    std::cout << "F-Matrix size= " << fundemental.rows << "," << fundemental.cols << std::endl;
    std::cout << "F-Matrix = \n" << fundemental << std::endl;

    // Draw the left points corresponding epipolar lines in right image
    std::vector<cv::Vec3f> lines1;
    cv::computeCorrespondEpilines(cv::Mat(selPoints1), 1,  fundemental, lines1);
    for (auto iter = lines1.begin(); iter != lines1.end(); ++iter){
        cv::line(image_right, cv::Point(0, -(*iter)[2] / (*iter)[1]),
                 cv::Point(image_right.cols, -((*iter)[2] + (*iter)[0] * image_right.cols) / (*iter)[1]),
                 cv::Scalar(255, 255, 255));
    }

    // Draw the RIGHT points corresponding epipolar lines in left image
    std::vector<cv::Vec3f> lines2;
    cv::computeCorrespondEpilines(cv::Mat(selPoints2), 2, fundemental, lines2);
    for (auto iter = lines2.begin(); iter != lines2.end(); ++iter){

        cv::line(image_left, cv::Point(0, -(*iter)[2] / (*iter)[1]),
                 cv::Point(image_left.cols, -((*iter)[2] + (*iter)[0] * image_left.cols) / (*iter)[1]),
                 cv::Scalar(255, 255, 255));
    }

    // Display the images with points and epipolar lines
    cv::imshow("Left Image", image_left);
    cv::waitKey();
    cv::imshow("Right Image", image_right);
    cv::waitKey();

    return 0;
}
