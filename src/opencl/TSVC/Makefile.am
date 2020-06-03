SUBDIRS = $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../common
#
AM_LDFLAGS = $(OCL_LDFLAGS)
AM_CPPFLAGS = $(OCL_CPPFLAGS)
#
# What is the destination for programs built from this directory?
serialopencldir = $(bindir)/Serial/OpenCL
#
# What programs should be installed in the bin/Serial/OpenCL destination?
serialopencl_PROGRAMS = S1119 S119 S124 S211FOURSTAGES S211TWOSTAGES S211SIXSTAGES

S1119_SOURCES		=	S1119.cpp main.cpp
S1119_LDADD		=	$(OCL_LIBS) $(LIBS)

S119_SOURCES		=	S119.cpp main.cpp
S119_LDADD		=	$(OCL_LIBS) $(LIBS)

S124_SOURCES		=	S124.cpp main.cpp
S124_LDADD		=	$(OCL_LIBS) $(LIBS)

S211TWOSTAGES_SOURCES	=	S211TWOSTAGES.cpp main.cpp
S211TWOSTAGES_LDADD	=	$(OCL_LIBS) $(LIBS)

S211FOURSTAGES_SOURCES	=	S211FOURSTAGES.cpp main.cpp
S211FOURSTAGES_LDADD	=	$(OCL_LIBS) $(LIBS)

S211SIXSTAGES_SOURCES	=	S211TWOSTAGES.cpp main.cpp
S211SIXSTAGES_LDADD	=	$(OCL_LIBS) $(LIBS)


