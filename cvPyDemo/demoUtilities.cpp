/* 
 * File:   demoUtilities.h
 * Author: mbustreo
 *
 * Created on 21 November 2015
 *
 * In this namespace are implemented some utility funtions 
 * used in CvPyMat::multiRotPersDet method.
 */

#include "demoUtilities.h"

#include <fstream>
#include <opencv2/imgproc.hpp>

using namespace cv;

namespace demo_utils
{
    double tt_tic = 0;

    void tic()
    {
        tt_tic = getTickCount();
    }

    void toc()
    {
        double tt_toc = (getTickCount() - tt_tic)/(getTickFrequency());
        printf ("toc: %4.3f sn\n", tt_toc);
    }

    Mat rotate(const Mat &src, double angle)
    {
        Mat dst;
        Point2f pt(src.cols/2., src.rows/2.);    
        Mat r = getRotationMatrix2D(pt, angle, 1.0);
        warpAffine(src, dst, r, Size(src.cols, src.rows));
        return dst;
    }

    void generateRotatedImages(const Mat &image, std::vector<Mat> &images, int degStep)
    {
        images.push_back(image);

        int rotVal = degStep;

        while(rotVal<360)
        {
            images.push_back(rotate(image, rotVal));
            rotVal += degStep;
        }
    }

    void drawBoxes(Mat &frame, std::vector<cv::dpm::DPMDetector::ObjectDetection> ds, float threshold, Scalar color)
    {        
        std::ofstream myfile;

        for (unsigned int i = 0; i < ds.size(); i++) 
        {

            if(ds[i].score<threshold)
                continue;

            rectangle(frame, ds[i].rect, color, 2);
            
            std::ostringstream ss;
            ss << ds[i].score;
            std::string score(ss.str());

            putText(frame, score , Point(ds[i].rect.x+10, ds[i].rect.y+10) , FONT_HERSHEY_SIMPLEX, 0.5, color);
        }
     }

    void saveDetections(int rot, std::vector<cv::dpm::DPMDetector::ObjectDetection> ds, std::ofstream &file, float threshold)
    {
        for (unsigned int i = 0; i < ds.size(); i++) 
        {
            if(ds[i].score < threshold)
                continue;

            char buffer [50];
            sprintf (buffer, "%d, %d, %d, %d, %d, %f\n", rot, ds[i].rect.x, ds[i].rect.y, ds[i].rect.height, ds[i].rect.width, ds[i].score);

            file << buffer;
        }
    }
}