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
serialopencl_PROGRAMS = Stream \
												Transpose \
												Random \
												Scatter \
												Reduction

Stream_SOURCES					=	Stream.cpp main.cpp
Stream_LDADD						= $(OCL_LIBS) $(LIBS)

Transpose_SOURCES				= Transpose.cpp main.cpp
Transpose_LDADD					= $(OCL_LIBS) $(LIBS)

Random_SOURCES					= Random.cpp main.cpp
Random_LDADD						= $(OCL_LIBS) $(LIBS)

Scatter_SOURCES					= Scatter.cpp main.cpp
Scatter_LDADD						= $(OCL_LIBS) $(LIBS)

Reduction_SOURCES				= Reduction.cpp main.cpp
Reduction_LDADD					= $(OCL_LIBS) $(LIBS)
