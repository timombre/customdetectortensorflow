CC ?= cc
CXX ?= c++

TF_BUILDDIR := /home/tmenais/tensorflow

CXXFLAGS := --std=c++11
INCLUDES := -I/usr/local/lib/python3.6/dist-packages/tensorflow/include/  -I$(TF_BUILDDIR) -I$(TF_BUILDDIR)/bazel-genfiles/ 
LIBS := -L$(TF_BUILDDIR)/bazel-bin/tensorflow -ltensorflow_cc -ltensorflow_framework -ldarknet

main: main.cc
	$(CXX) customdetector.cpp $(CXXFLAGS) $(INCLUDES) $(LIBS) -o customdetector

clean:
	rm -f customdetector
