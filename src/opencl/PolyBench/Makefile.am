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

serialopencldir = $(bindir)/Serial/OpenCL

serialopencl_PROGRAMS = 2mm \
												mm1

2mm_SOURCES					= 2mm.cpp main.cpp
2mm_LDADD						= $(OCL_LIBS) $(LIBS)

mm1_SOURCES					= mm1.cpp main.cpp
mm1_LDADD						= $(OCL_LIBS) $(LIBS)
