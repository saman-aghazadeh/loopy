if BUILD_MPI
	MAYBE_MPI = epmpi
endif

SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../common

AM_LDFLAGS = $(OCL_LDFLAGS)
AM_CPPFLAGS = $(OCL_CPPFLAGS)

# What is the destination for programs built from this directory?
serialopencldir = $(bindir)/Serial/OpenCL

# What programs should be installed in the bin/Serial/OpenCL destination?
serialopencl_PROGRAMS = Flops \
												PerfMeasure \
												AlgToCode

Flops_SOURCES	= Flops.cpp main.cpp
Flops_LDADD		= $(OCL_LIBS) $(LIBS)

PerfMeasure_SOURCES	=	PerfMeasure.cpp main.cpp
PerfMeasure_LDADD		= $(OCL_LIBS) $(LIBS)

AlgToCode_SOURCES	=	AlgToCode.cpp main.cpp
AlgToCode_LDADD		=	$(OCL_LIBS) $(LIBS)
