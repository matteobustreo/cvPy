PYTHON_INC=$(shell python2-config --includes)
PYTHON_LIBS=$(shell python2-config --libs)
BOOST_LIBS=-lboost_python

export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig

all: CvPyMat.cpp CvPyMat.h demoUtilities.cpp demoUtilities.h
	g++ -O3 -I/usr/include/python2.7 -I/usr/local/lib/python2.7/dist-packages/numpy/core/include/ -shared -fPIC `pkg-config --cflags opencv` CvPyMat.cpp demoUtilities.cpp `pkg-config --libs opencv` -o CvPyMat.so $(PYTHON_INC) $(PYTHON_LIBS) $(BOOST_LIBS) 
