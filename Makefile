CC ?= cc
CXX ?= c++

TF_BUILDDIR := /home/tmenais/tensorflow

CXXFLAGS := --std=c++11 
INCLUDES := -I/usr/local/lib/python3.6/dist-packages/tensorflow/include/  -I$(TF_BUILDDIR) -I$(TF_BUILDDIR)/bazel-genfiles/ 
LIBS := -L$(TF_BUILDDIR)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_objdetect -lopencv_videoio -lopencv_imgcodecs -g -lm

main: main.cc
	$(CXX) customdetectoropencv.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o customdetectoropencv

clean:
	rm -f customdetectoropencv
