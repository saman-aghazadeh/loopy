if BUILD_MPI
	MAYBE_MPI = epmpi
endif

#SUBDIRS	= $(MAYBE_MPI)

include $(top_builddir)/config/config.mk
include $(top_builddir)/config/targets.mk

# How to find source files
VPATH = $(srcdir):$(srcdir)/../common:$(srcdir)/Algs

AM_LDFLAGS = $(CUDA_LDFLAGS)
AM_CPPFLAGS = $(CUDA_INC)

# What is the destination for programs built from this directory?
serialcudadir = $(bindir)/Serial/CUDA

# What programs should be installed in the bin/Serial/CUDA destination?
serialcuda_PROGRAMS = AlgToCode

AlgToCode_SOURCES				= main.cpp
AlgToCode_LDADD					= AlgToCode.o $(CUDA_LIBS) $(LIBS)

