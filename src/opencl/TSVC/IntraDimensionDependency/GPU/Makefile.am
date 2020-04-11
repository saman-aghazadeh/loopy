# if BUILD_MPI
#     MAYBE_MPI = epmpi
# endif

SUBDIRS	= $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../../../common

AM_LDFLAGS = $(OCL_LDFLAGS)
AM_CPPFLAGS = $(OCL_CPPFLAGS)

# what is the destination for programs built from this directory?
serialopencldir = $(bindir)/Serial/OpenCL

# what programs should be installed in the bin/Serial/OpenCL destination?
serialopencl_PROGRAMS = IntraDimensionDependency

IntraDimensionDependency_SOURCES	= IntraDimensionDependency.cpp main.cpp
IntraDimensionDependency_LDADD		= $(OCL_LIBS) $(LIBS)
