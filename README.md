cvPy
======

**cvPy** is a simple C++ code for converting OpenCV Mat objects to Python and easily reusing OpenCV C++ code in Python. It is a personal adaptation of CvBridge - module_opencv2.cpp functions (http://wiki.ros.org/cv_bridge).

## Usage
cvPy have been tested under Manjaro 15.09 with OpenCV3.0 and Python 2.7. cvPy requires Numpy and Boost.Python.  
See pyttest for some sample uses.

#### Dockerfile
Using the included Dockerfile you can automatically build an image with the required packages using ubuntu:14.04 as base image.

Build the images using 
```
docker build -t cvpydemo . 
```
 
Run the images using 
```
docker run -i -t cvpydemo
```
 
You can find the project in ```/Project/CvPyDemo``` and you can test it with
```
cd /Project/CvPyDemo
python pyTest.py
```

Output result will be written in ```/Project/CvPyDemo/detectorResults.csv```

## See also
[www.matteobustreo.com](http://www.matteobustreo.com)