FROM ubuntu:14.04

# Installing all the packages required by OpenCV
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
	build-essential  	\
	cmake  				\
	git  				\
	pkg-config 			\

	libjpeg8-dev  		\
	libtiff4-dev  		\
	libjasper-dev  		\
	libpng12-dev  		\
	
	libgtk2.0-dev  		\

	libavcodec-dev  	\
	libavformat-dev  	\
	libswscale-dev  	\
	libv4l-dev  		\

	libatlas-base-dev 	\
	gfortran  			\

	wget  				\
	python2.7-dev  		\

	libboost-all-dev  	\
	locate  



# Installing numpy using pip
RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install numpy

# Downloading and installing OpenCV and Extra Modules
RUN cd ~ && git clone https://github.com/Itseez/opencv.git
RUN cd ~ && git clone https://github.com/Itseez/opencv_contrib.git

# Compiling OpenCV with Extra Modules
RUN cd ~/opencv && mkdir ./build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_EXAMPLES=OFF -D BUILD_opencv_xfeatures2d=OFF -D BUILD_opencv_stereo=OFF -D BUILD_opencv_aruco=OFF -D BUILD_opencv_tracking=OFF -D BUILD_opencv_ximgproc=OFF .. \
	&& make \
	&& make install

# Creating the symbolic links for the OpenCV shared libraries
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf && ldconfig -v

# Installing X11 display server
RUN apt-get update && apt-get upgrade -y && apt-get install -y xvfb

# Set a virtual monitor to autostart when starting the container
RUN sed -i -e '$a\Xvfb :1 -screen 0 1024x768x16 &> xvfb.log & DISPLAY=:1.0 && export DISPLAY' /etc/bash.bashrc

# Copying and compiling demo project
RUN mkdir Project
COPY ./cvPyDemo/* /Project/CvPyDemo/
RUN cd /Project/CvPyDemo && make