// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn
// 
// monofonIC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// monofonIC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
#pragma once

#include <logger.hh>

#include <complex>
#include <map>
#include <memory>

#if defined(USE_MPI)
#include <mpi.h>
#include <fftw3-mpi.h>
#else
#include <fftw3.h>
#endif

#include "config_file.hh"

//! use to suppress warnings of unused variables
#define _unused(x) ((void)(x))

//! assert on all elements of a brace enclosed initializer list (useful for variadic templates)
inline void list_assert_all( const std::initializer_list<bool>& t )
{
  for( auto b : t ) {assert(b);_unused(b);}
}

// include CMake controlled configuration settings
#include "cmake_config.hh"

#if defined(USE_PRECISION_FLOAT)
using real_t = float;
using complex_t = fftwf_complex;
#define FFTW_PREFIX fftwf
#elif defined(USE_PRECISION_DOUBLE)
using real_t = double;
using complex_t = fftw_complex;
#define FFTW_PREFIX fftw
#elif defined(USE_PRECISION_LONGDOUBLE)
using real_t = long double;
using complex_t = fftwl_complex;
#define FFTW_PREFIX fftwl
#endif

enum class fluid_component
{
  density,
  vx,
  vy,
  vz,
  dx,
  dy,
  dz,
  mass
};
enum class cosmo_species
{
  dm,
  baryon,
  neutrino
};
extern std::map<cosmo_species, std::string> cosmo_species_name;

using ccomplex_t = std::complex<real_t>;

#define FFTW_GEN_NAME_PRIM(a, b) a##_##b
#define FFTW_GEN_NAME(a, b) FFTW_GEN_NAME_PRIM(a, b)
#define FFTW_API(x) FFTW_GEN_NAME(FFTW_PREFIX, x)

using fftw_plan_t = FFTW_GEN_NAME(FFTW_PREFIX, plan);

#if defined(FFTW_MODE_PATIENT)
#define FFTW_RUNMODE FFTW_PATIENT
#elif defined(FFTW_MODE_MEASURE)
#define FFTW_RUNMODE FFTW_MEASURE
#else
#define FFTW_RUNMODE FFTW_ESTIMATE
#endif

#if defined(USE_MPI)
inline double get_wtime()
{
  return MPI_Wtime();
}

namespace MPI
{

inline int get_rank(void)
{
  int rank, ret;
  ret = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  assert(ret == MPI_SUCCESS);
  _unused(ret);
  return rank;
}

inline int get_size(void)
{
  int size, ret;
  ret = MPI_Comm_size(MPI_COMM_WORLD, &size);
  assert(ret == MPI_SUCCESS);
  _unused(ret);
  return size;
}

template <typename T>
inline MPI_Datatype get_datatype(void)
{
  if (typeid(T) == typeid(std::complex<float>))
    return MPI_C_FLOAT_COMPLEX;

  if (typeid(T) == typeid(std::complex<double>))
    return MPI_C_DOUBLE_COMPLEX;

  if (typeid(T) == typeid(std::complex<long double>))
    return MPI_C_LONG_DOUBLE_COMPLEX;

  if (typeid(T) == typeid(int))
    return MPI_INT;

  if (typeid(T) == typeid(unsigned))
    return MPI_UNSIGNED;

  if (typeid(T) == typeid(float))
    return MPI_FLOAT;

  if (typeid(T) == typeid(double))
    return MPI_DOUBLE;

  if (typeid(T) == typeid(long double))
    return MPI_LONG_DOUBLE;

  if (typeid(T) == typeid(char))
    return MPI_CHAR;

  abort();
}

inline std::string get_version(void)
{
  int len;
  char mpi_lib_ver[MPI_MAX_LIBRARY_VERSION_STRING];

  MPI_Get_library_version(mpi_lib_ver, &len);
  return std::string(mpi_lib_ver);
}
} // namespace MPI

#else
#if defined(_OPENMP)
#include <omp.h>
inline double get_wtime()
{
  return omp_get_wtime();
}
#else
#include <ctime>
inline double get_wtime()
{
  return std::clock() / double(CLOCKS_PER_SEC);
}
#endif
#endif

inline void multitask_sync_barrier(void)
{
#if defined(USE_MPI)
  MPI_Barrier(MPI_COMM_WORLD);
#endif
}

extern size_t global_mem_high_mark, local_mem_high_mark;

namespace CONFIG
{
extern int MPI_thread_support;
extern int MPI_task_rank;
extern int MPI_task_size;
extern bool MPI_ok;
extern bool MPI_threads_ok;
extern bool FFTW_threads_ok;
extern int num_threads;
} // namespace CONFIG