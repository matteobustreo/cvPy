/* 
 * File:   demoUtilities.h
 * Author: mbustreo
 *
 * Created on 21 November 2015
 *
 * In this namespace are implemented some utility funtions 
 * used in CvPyMat::multiRotPersDet method.
 */

#ifndef DEMO_UTILITIES_H
#define DEMO_UTILITIES_H

#include <opencv2/dpm.hpp>
#include <opencv2/core/mat.hpp>

namespace demo_utils
{
	// Timing Utilities
	void tic();
	void toc();

	// Rotation Utilities
	cv::Mat  rotate(const cv::Mat &src, double angle);
	void generateRotatedImages(const cv::Mat &image, std::vector<cv::Mat> &images, int degStep);

	// Drawing Utilities
	void drawBoxes(cv::Mat &frame, std::vector<cv::dpm::DPMDetector::ObjectDetection> ds, float threshold, cv::Scalar color);

	// Saving Utilities
	void saveDetections(int rot, std::vector<cv::dpm::DPMDetector::ObjectDetection> ds, std::ofstream &file, float threshold);
}

#endif  /* DEMO_UTILITIES_H */

    