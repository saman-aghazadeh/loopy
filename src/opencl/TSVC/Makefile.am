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

serialopencl_PROGRAMS = S000 \
												S111

S000_SOURCES					= S000.cpp main.cpp
S000_LDADD						= $(OCL_LIBS) $(LIBS)

S111_SOURCES					= S111.cpp main.cpp
S111_LDADD						= $(OCL_LIBS) $(LIBS)
