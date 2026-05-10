#/******************************************************************************
# * Copyright (c) 2011, Duane Merrill.  All rights reserved.
# * Copyright (c) 2011-2016, NVIDIA CORPORATION.  All rights reserved.
# * 
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *     * Redistributions of source code must retain the above copyright
# *       notice, this list of conditions and the following disclaimer.
# *     * Redistributions in binary form must reproduce the above copyright
# *       notice, this list of conditions and the following disclaimer in the
# *       documentation and/or other materials provided with the distribution.
# *     * Neither the name of the NVIDIA CORPORATION nor the
# *       names of its contributors may be used to endorse or promote products
# *       derived from this software without specific prior written permission.
# * 
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
# * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# *
#******************************************************************************/

#-------------------------------------------------------------------------------
#
# Makefile usage
#
# CPU:
#   make cpu_spmv
#
# GPU:
#   make gpu_spmv [sm=<XXX,...>] [verbose=<0|1>]
#
# Examples:
#   make gpu_spmv sm=750        # Colab T4
#   make gpu_spmv sm=800        # Colab A100
#   make gpu_spmv sm=700        # V100
#   make gpu_spmv sm=890        # L4
#   make gpu_spmv sm=350        # Original K40 (paper hardware)
#
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# Commandline Options
#-------------------------------------------------------------------------------

# [sm=<XXX,...>] Compute-capability to compile for
# Accepts comma-separated list, e.g., sm=750,800

COMMA = ,
ifdef sm
    SM_ARCH = $(subst $(COMMA),-,$(sm))
else
    SM_ARCH = 750   # DEFAULT: T4 (most common Colab GPU)
endif

# ---- Original architectures (kept for backwards compatibility) ----
ifeq (520, $(findstring 520, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_52,code="sm_52,compute_52"
endif
ifeq (370, $(findstring 370, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_37,code="sm_37,compute_37"
endif
ifeq (350, $(findstring 350, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_35,code="sm_35,compute_35"
endif
ifeq (300, $(findstring 300, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_30,code="sm_30,compute_30"
endif

# ---- Modern architectures (added for Colab / current hardware) ----
ifeq (700, $(findstring 700, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_70,code="sm_70,compute_70"   # V100
endif
ifeq (720, $(findstring 720, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_72,code="sm_72,compute_72"   # Xavier
endif
ifeq (750, $(findstring 750, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_75,code="sm_75,compute_75"   # T4 (Colab default)
endif
ifeq (800, $(findstring 800, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_80,code="sm_80,compute_80"   # A100
endif
ifeq (860, $(findstring 860, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_86,code="sm_86,compute_86"   # RTX 30xx
endif
ifeq (870, $(findstring 870, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_87,code="sm_87,compute_87"   # Orin
endif
ifeq (890, $(findstring 890, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_89,code="sm_89,compute_89"   # L4 / RTX 40xx
endif
ifeq (900, $(findstring 900, $(SM_ARCH)))
    SM_TARGETS += -gencode=arch=compute_90,code="sm_90,compute_90"   # H100
endif


# [verbose=<0|1>] Verbose toolchain output
ifeq ($(verbose), 1)
    NVCCFLAGS += -v
endif


#-------------------------------------------------------------------------------
# Compiler Detection
#-------------------------------------------------------------------------------

CUB_DIR = $(dir $(lastword $(MAKEFILE_LIST)))

NVCC = "$(shell which nvcc)"
ifdef nvccver
    NVCC_VERSION = $(nvccver)
else
    NVCC_VERSION = $(strip $(shell nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//'))
endif

# Detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])


#-------------------------------------------------------------------------------
# NVCC Flags
#-------------------------------------------------------------------------------

# Removed: -Xptxas -v -Xcudafe -#
# These caused errors on CUDA 11+ / 12+. Safe to remove — they were
# just for printing register/smem usage and are not needed for correctness.

NVCCFLAGS += $(SM_DEF)

ifeq (WIN_NT, $(findstring WIN_NT, $(OSUPPER)))
    NVCCFLAGS += -Xcompiler /fp:strict
    NVCCFLAGS += -Xcompiler /bigobj -Xcompiler /Zm500
    CC = cl
    NPPI = -lnppi
    NVCCFLAGS += -Xcompiler /MT
ifneq ($(force32), 1)
    CUDART_CYG = "$(shell dirname $(NVCC))/../lib/Win32/cudart.lib"
else
    CUDART_CYG = "$(shell dirname $(NVCC))/../lib/x64/cudart.lib"
endif
    CUDART = "$(shell cygpath -w $(CUDART_CYG))"
else
    # Linux / Colab
    NVCCFLAGS += -Xcompiler -ffloat-store
    CC = g++
ifneq ($(force32), 1)
    CUDART = "$(shell dirname $(NVCC))/../lib/libcudart_static.a"
else
    CUDART = "$(shell dirname $(NVCC))/../lib64/libcudart_static.a"
endif
endif


#-------------------------------------------------------------------------------
# CPU Compiler — g++ with OpenMP (replaces icpc + MKL)
#
# ORIGINAL (Intel only — does NOT work on Colab):
#   OMPCC      = icpc
#   OMPCC_FLAGS= -openmp -O3 -lrt -fno-alias -xHost -lnuma -O3 -mkl
#
# UPDATED: uses g++ + OpenMP, no MKL dependency
#-------------------------------------------------------------------------------

OMPCC       = g++
OMPCC_FLAGS = -fopenmp -O3 -march=native -std=c++14 -lrt


#-------------------------------------------------------------------------------
# Includes & Dependencies
#-------------------------------------------------------------------------------

INC += -I$(CUB_DIR) -I$(CUB_DIR)test

rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *,%,$2),$d))

DEPS =  $(call rwildcard, $(CUB_DIR),*.cuh) \
        $(call rwildcard, $(CUB_DIR),*.h) \
        Makefile


#-------------------------------------------------------------------------------
# Targets
#-------------------------------------------------------------------------------

.PHONY: all clean gpu_spmv cpu_spmv

all: gpu_spmv cpu_spmv


#-------------------------------------------------------------------------------
# make clean
#-------------------------------------------------------------------------------

clean:
	rm -f _gpu_spmv_driver _cpu_spmv_driver


#-------------------------------------------------------------------------------
# make gpu_spmv
#
# Builds the GPU (CUDA) merge-based SpMV binary.
# Output: ./_gpu_spmv_driver
#
# Usage:
#   make gpu_spmv sm=750      # T4
#   make gpu_spmv sm=800      # A100
#-------------------------------------------------------------------------------

gpu_spmv: gpu_spmv.cu $(DEPS)
	$(NVCC) $(DEFINES) $(SM_TARGETS) \
	    -o _gpu_spmv_driver gpu_spmv.cu \
	    $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) \
	    -lcusparse -O3


#-------------------------------------------------------------------------------
# make cpu_spmv
#
# Builds the CPU (OpenMP) merge-based SpMV binary.
# Output: ./_cpu_spmv_driver
#
# Note: MKL comparison (Intel MKL CsrMV) is disabled because icpc/MKL
#       are not available on Colab. The merge-path OpenMP implementation
#       still builds and runs correctly without MKL.
#       To re-enable MKL: install Intel oneAPI and switch back to icpc.
#-------------------------------------------------------------------------------

cpu_spmv: cpu_spmv.cpp $(DEPS)
	$(OMPCC) $(DEFINES) \
	    -o _cpu_spmv_driver cpu_spmv.cpp \
	    $(OMPCC_FLAGS) $(INC)