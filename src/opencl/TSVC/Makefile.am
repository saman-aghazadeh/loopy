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
												S111 \
												S1111 \
												S112 \
												S1112 \
												S113 \
												S119 \
												S1119 \
												S122 \
												S124 \
												S176 \
												S211

S000_SOURCES					= S000.cpp main.cpp
S000_LDADD						= $(OCL_LIBS) $(LIBS)

S111_SOURCES					= S111.cpp main.cpp
S111_LDADD						= $(OCL_LIBS) $(LIBS)

S1111_SOURCES					= S1111.cpp main.cpp
S1111_LDADD						= $(OCL_LIBS) $(LIBS)

S112_SOURCES					= S112.cpp main.cpp
S112_LDADD						= $(OCL_LIBS) $(LIBS)

S1112_SOURCES					= S1112.cpp main.cpp
S1112_LDADD						= $(OCL_LIBS) $(LIBS)

S113_SOURCES					= S113.cpp main.cpp
S113_LDADD						= $(OCL_LIBS) $(LIBS)

S119_SOURCES					= S119.cpp main.cpp
S119_LDADD						= $(OCL_LIBS) $(LIBS)

S1119_SOURCES					= S1119.cpp main.cpp
S1119_LDADD						= $(OCL_LIBS) $(LIBS)

S122_SOURCES					= S122.cpp main.cpp
S122_LDADD						= $(OCL_LIBS) $(LIBS)

S124_SOURCES					= S124.cpp main.cpp
S124_LDADD						= $(OCL_LIBS) $(LIBS)

S176_SOURCES					= S176.cpp main.cpp
S176_LDADD						= $(OCL_LIBS) $(LIBS)

S211_SOURCES					= S211.cpp main.cpp
S211_LDADD						= $(OCL_LIBS) $(LIBS)
