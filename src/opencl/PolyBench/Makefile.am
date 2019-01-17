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

serialopencl_PROGRAMS = mm2 \
												mm1 \
												doitgen \
												FloydWarshall

mm2_SOURCES					= mm2.cpp main.cpp
mm2_LDADD						= $(OCL_LIBS) $(LIBS)

mm1_SOURCES					= mm1.cpp main.cpp
mm1_LDADD						= $(OCL_LIBS) $(LIBS)

doitgen_SOURCES			= doitgen.cpp main.cpp
doitgen_LDADD				= $(OCL_LIBS) $(LIBS)

FloydWarshall_SOURCES	= FloydWarshall.cpp main.cpp
FloydWarshall_LDADD		= $(OCL_LIBS) $(LIBS)
