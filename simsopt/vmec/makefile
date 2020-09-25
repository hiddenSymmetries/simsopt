VMEC_PATH ?= ${STELLOPT_PATH}/VMEC2000
include ${STELLOPT_PATH}/make.inc
MAKE = makevmec2000_py

#=======================================================================
#                   define the compiler names
#=======================================================================

CC       = gcc
FC       = ${MPI_COMPILE}
F90      = gfortran
PYTHON   = python

#=======================================================================
#                     additional flags
#=======================================================================

ifeq ($(F90),gfortran)
	FPP      = $(FC) -E
	FPP_F90FLAGS = -x f95-cpp-input -fPIC
	F90FLAGS = $(FLAGS_R)
    FCOMP    = $(MPI_COMPILE) #gfortran
endif

# Settings for Matt's macbook:
# Because MacOS does not let you set LD_LIBRARY_PATH, a critical step was 
# sudo ln -s ~/stellopt_github/develop/STELLOPT/bin/libstell.so /usr/local/lib
ifeq ($(SFINCS_SYSTEM),macports)
  CC = mpicc
  FC = mpif90
  F90 = mpif90
  FPP = mpif90 -E
  FPP_F90FLAGS = -x f95-cpp-input -fPIC
  FCOMP = mpif90
endif

CFLAGS = -fPIC #     ==> universal for ifort, gfortran, pgi

#=======================================================================
#=======================================================================

UNAME = $(shell uname)

ifeq (${UNAME}, Darwin)
  LIBTOOL = libtool -static -o
else
  LIBTOOL = ar src
endif

# ======================================================================
# PROJECT CONFIG, do not put spaced behind the variables
# ======================================================================
# Python module name
PYTHON_MODN = vmec_f90wrap
# mapping between Fortran and C types
KIND_MAP = kind_map
# static library name
CODE_NAME = vmec
#LIB_NAME = lib$(PYTHON_MODN).a
LIB_NAME = lib$(CODE_NAME).a

ADDITIONAL_LINKS = -L${STELLOPT_PATH}/LIBSTELL/Release -lstell ${LIB_SHARE} ${LIBS}
ADDITIONAL_INCLUDES = -I${VMEC_PATH}/Release -I${STELLOPT_PATH}/LIBSTELL/Release ${MOD1_PATH}
#=======================================================================
#       List all source files that require a Python interface
#=======================================================================

# files to be wrapped
LIBSRC_WRAP_SOURCES := runvmec.f \
		    reinit.f90 \
		    vmec_input.f \
		    read_wout_mod.f90
# object files
LIBSRC_WRAP_OBJECTS = $(patsubst %.f90,%.o,$(patsubst %.f,%.o, ${LIBSRC_WRAP_SOURCES}))

# fpp files
LIBSRC_WRAP_FPP_FILES = $(LIBSRC_WRAP_OBJECTS:.o=.fpp)

#=======================================================================
#                 Relevant suffixes
#=======================================================================

.SUFFIXES: .f90 .fpp

#=======================================================================
#
#=======================================================================

.PHONY: all all_clean

all: _${PYTHON_MODN}.so test

f90wrap_clean:
	-rm -f ${LIBSRC_OBJECTS} ${LIBSRC_FPP_FILES} $(LIB_NAME) _${PYTHON_MODN}*.so \
	_${PYTHON_MODN}_pkg.so *.mod *.fpp f90wrap*.f90 f90wrap*.o *.o ${PYTHON_MODN}.py
	-rm -rf ${PYTHON_MODN}_pkg
	-rm -rf src.*/ .f2py_f2cmap .libs/ __pycache__/
	-rm -f $(LIB_NAME)
	-rm -f ${VMEC_PATH}/Release/reinit.f

all_clean: f90wrap_clean
	-rm -f ${VMEC_PATH}/Release/*.o
	-rm -f ${VMEC_PATH}/Release/*.mod

%.fpp : %.f
	${FPP} ${FPP_F90FLAGS} $< -o $@
%.fpp : %.f90
	${FPP} ${FPP_F90FLAGS} $< -o $@

$(LIB_NAME): ${LIBSRC_WRAP_SOURCES}
	@echo "MAKING STATIC (RELEASE) EXECUTABLE"
	@cp ${MAKE} ${VMEC_PATH}/Release/. ; cp reinit.f90 ${VMEC_PATH}/Release/.
	@cd ${VMEC_PATH}/Release; make -f $(MAKE) $(LIB_NAME) \
	FLAGS="$(FLAGS_R) $(MOD1_PATH)" SFLAGS="" TYPE="release" LOCTYPE="Release"; rm -f ${MAKE}
	@mv ${VMEC_PATH}/Release/${LIB_NAME} .; rm ${VMEC_PATH}/Release/reinit.f90 ; rm ${VMEC_PATH}/Release/reinit.o
	@echo "Static Release version - ${LIB_NAME} - is now updated"
	ranlib $@

_${PYTHON_MODN}.so: $(LIB_NAME) ${LIBSRC_WRAP_FPP_FILES} 
	f90wrap -m ${PYTHON_MODN} ${LIBSRC_WRAP_FPP_FILES} -k ${KIND_MAP} -v
	f2py-f90wrap --f90exec=${FC} --build-dir . -c -m _${PYTHON_MODN} \
	-L. -l$(CODE_NAME) ${ADDITIONAL_LINKS} -I. ${ADDITIONAL_INCLUDES} f90wrap*.f90	

test:
	@python -c "import $(PYTHON_MODN); print('import successfully')"

test_make:
	@echo 'LIBSRC_WRAP_SOURCES = $(LIBSRC_WRAP_SOURCES)'
	@echo 'LIBSRC_WRAP_OBJECTS = $(LIBSRC_WRAP_OBJECTS)'
	@echo 'LIBSRC_WRAP_FPP_FILES = $(LIBSRC_WRAP_FPP_FILES)'
	@echo 'ADDITIONAL_LINKS = ${ADDITIONAL_LINKS}'
	@echo 'ADDITIONAL_INCLUDES = ${ADDITIONAL_INCLUDES}'
