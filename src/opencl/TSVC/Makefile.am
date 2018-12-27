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

serialopencl_PROGRAMS = S1119 \
												S122 \
												S124 \
												S176 \
												S211 \
												S211TWOSTAGES \
												S211FOURSTAGES \
												S211SIXSTAGES \
												S119WAVEFRONT \
												S221

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

S211FOURSTAGES_SOURCES	=	S211FOURSTAGES.cpp main.cpp
S211FOURSTAGES_LDADD		=	$(OCL_LIBS) $(LIBS)

S211TWOSTAGES_SOURCES		= S211TWOSTAGES.cpp main.cpp
S211TWOSTAGES_LDADD			= $(OCL_LIBS) $(LIBS)

S211SIXSTAGES_SOURCES		= S211SIXSTAGES.cpp main.cpp
S211SIXSTAGES_LDADD			= $(OCL_LIBS) $(LIBS)

S221_SOURCES					= S221.cpp main.cpp
S221_LDADD						= $(OCL_LIBS) $(LIBS)

S119WAVEFRONT_SOURCES	= S119-wavefront.cpp main.cpp
S119WAVEFRONT_LDADD		= $(OCL_LIBS) $(LIBS)
