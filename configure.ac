# initial information about the project
AC_INIT([dsmh-npsol],[1.0],[hongwei.wu@atl.frb.org])

# macro directory
AC_CONFIG_MACRO_DIRS([m4])

# check if the source folder is available
AC_CONFIG_SRCDIR([src/dsmh_basic/CEquiEnergyModel.cpp])
AC_CONFIG_FILES([data/Makefile])

# check for C++ preprocessor and compiler
#AC_PROG_CXXCPP
#AC_PROG_CXX
AC_PROG_CC

# C++11 standard
#AX_CXX_COMPILE_STDCXX_11

# automake initialization (mandatory) including a check for automake API version >= 1.9
AM_INIT_AUTOMAKE([1.9])

# use the C++ compiler for the following checks
#AC_LANG([C++])
AC_LANG([C])

# Checks for header files.
#AC_HEADER_STDC
#AC_CHECK_HEADERS([algorithm cmath cstdio cstdlib ctime errno.h fcntl.h fstream functional glob.h iomanip iostream math.h sstream string string.h sys/file.h sys/stat.h sys/types.h time.h unistd.h vector])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# mpi
sinclude(m4_ax_prog_cxx_mpi.m4)
AX_PROG_CXX_MPI([with_mpi=yes],[],[AC_MSG_ERROR([mpi compiler requested but not found.])])

# -lm, -lgslcblas, -lgsl 
AC_CHECK_LIB([m],[cos])
AC_CHECK_LIB([gslcblas],[cblas_dgemm])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm])

# blas lapack
sinclude(m4_ax_lapack.m4)
AX_LAPACK([LIBS="$LAPACK_LIBS $BLAS_LIBS $FCLIBS"],[AC_MSG_ERROR([blas/lapack requested but not found.])])
#LIBS="$LAPACK_LIBS $BLAS_LIBS $FCLIBS"

# npsol
#AC_FC_FUNC(npoptn)
#AC_CHECK_LIB([npsol_f77],[$npoptn],[],[echo "libnpsol_f77.a not found"])
#LIBS="-lnpsol_f77 -llssol_f77 $(LIBS)"

# distribute additional compiler and linker flags among Makefiles
# --> set and change these variables instead of CXXFLAGS or LDFLAGS (for user only)
AC_SUBST([LIBS])
AC_SUBST([AM_CXXFLAGS])
AC_SUBST([AM_LDFLAGS])

# files to generate via autotools (prepare .am or .in source files)
AC_CONFIG_FILES([Makefile])
AC_CONFIG_FILES([src/Makefile])

# finally this generates the Makefiles etc. for the build
AC_OUTPUT
