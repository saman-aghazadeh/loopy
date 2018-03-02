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
serialopencl_PROGRAMS = AlgToCode \
												AlgTypeGeneration \
												AlgToCodeOps \
												AlgToCodeCUs \
												AlgToCodeDepDegree \
												AlgToCodeDepDegreeST \
												AlgToCodeTwoLoops \
												AlgToCodeInterleaving \
												AlgToCodeUnrollTest \
												AlgToCodeUnfolding \
												AlgToCodeInterleaving2 \
												AlgToCodeUnfolding2 \
												AlgToCodeUnfolding3 \
											  AlgToCodeMatrixMultiplicationPipeline

AlgToCode_SOURCES								=	AlgToCode.cpp main.cpp
AlgToCode_LDADD									=	$(OCL_LIBS) $(LIBS)

AlgTypeGeneration_SOURCES 			= AlgTypeGeneration.cpp main.cpp
AlgTypeGeneration_LDADD					= $(OCL_LIBS) $(LIBS)

AlgToCodeOps_SOURCES						= AlgToCodeOps.cpp main.cpp
AlgToCodeOps_LDADD							= $(OCL_LIBS) $(LIBS)

AlgToCodeCUs_SOURCES						= AlgToCodeCUs.cpp main.cpp
AlgToCodeCUs_LDADD							= $(OCL_LIBS) $(LIBS)

AlgToCodeDepDegree_SOURCES			= AlgToCodeDepDegree.cpp main.cpp
AlgToCodeDepDegree_LDADD				= $(OCL_LIBS) $(LIBS)

AlgToCodeDepDegreeST_SOURCES		= AlgToCodeDepDegreeST.cpp main.cpp
AlgToCodeDepDegreeST_LDADD			=	$(OCL_LIBS) $(LIBS)

AlgToCodeTwoLoops_SOURCES				=	AlgToCodeTwoLoops.cpp main.cpp
AlgToCodeTwoLoops_LDADD					= $(OCL_LIBS) $(LIBS)

AlgToCodeInterleaving_SOURCES		=	AlgToCodeInterleaving.cpp main.cpp
AlgToCodeInterleaving_LDADD			= $(OCL_LIBS) $(LIBS)

AlgToCodeUnrollTest_SOURCES			= AlgToCodeUnrollTest.cpp main.cpp
AlgToCodeUnrollTest_LDADD				= $(OCL_LIBS) $(LIBS)

AlgToCodeUnfolding_SOURCES			= AlgToCodeUnfolding.cpp main.cpp
AlgToCodeUnfolding_LDADD				= $(OCL_LIBS) $(LIBS)

AlgToCodeInterleaving2_SOURCES	= AlgToCodeInterleaving2.cpp main.cpp
AlgToCodeInterleaving2_LDADD		= $(OCL_LIBS) $(LIBS)

AlgToCodeUnfolding2_SOURCES			= AlgToCodeUnfolding2.cpp main.cpp
AlgToCodeUnfolding2_LDADD				= $(OCL_LIBS) $(LIBS)

AlgToCodeUnfolding3_SOURCES			= AlgToCodeUnfolding3.cpp main.cpp
AlgToCodeUnfolding3_LDADD				= $(OCL_LIBS) $(LIBS)

AlgToCodeMatrixMultiplicationPipeline_SOURCES			= AlgToCodeMatrixMultiplicationPipeline.cpp main.cpp
AlgToCodeMatrixMultiplicationPipeline_LDADD				= $(OCL_LIBS) $(LIBS)
