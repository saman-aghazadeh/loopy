include ${top_builddir}/config/common.mk
include ${top_builddir}/config/targets.mk

VPATH = $(srcdir):$(srcdir)/../common

noinst_LIBRARIES	= libSHOCCommonOpenCL.a
libSHOCCommonOpenCL_a_SOURCES = OpenCLDeviceInfo.cpp \
	OpenCLPlatform.cpp \
	OpenCLNodePlatformContainer.cpp \
	Event.cpp \
	opencl.cpp \
  options.cpp \
	OpenCLEngine.cpp
