/*******************************************************************\
 testing.hh - This file is part of MUSIC2 -
 a code to generate initial conditions for cosmological simulations 
 
 CHANGELOG (only majors, for details see repo):
    10/2019 - Michael Michaux & Oliver Hahn - first implementation
\*******************************************************************/
#pragma once

#include <array>
#include <general.hh>
#include <config_file.hh>
#include <grid_fft.hh>
#include <cosmology_calculator.hh>

namespace testing{
    void output_potentials_and_densities( 
        ConfigFile& the_config,
        size_t ngrid, real_t boxlen,
        Grid_FFT<real_t>& phi,
        Grid_FFT<real_t>& phi2,
        Grid_FFT<real_t>& phi3a,
        Grid_FFT<real_t>& phi3b,
        std::array< Grid_FFT<real_t>*,3 >& A3 );

    void output_velocity_displacement_symmetries(
        ConfigFile &the_config,
        size_t ngrid, real_t boxlen, real_t vfac, real_t dplus,
        Grid_FFT<real_t> &phi,
        Grid_FFT<real_t> &phi2,
        Grid_FFT<real_t> &phi3a,
        Grid_FFT<real_t> &phi3b,
        std::array<Grid_FFT<real_t> *, 3> &A3,
        bool bwrite_out_fields=false);

    void output_convergence(
        ConfigFile &the_config,
        CosmologyCalculator* the_cosmo_calc,
        std::size_t ngrid, real_t boxlen, real_t vfac, real_t dplus,
        Grid_FFT<real_t> &phi,
        Grid_FFT<real_t> &phi2,
        Grid_FFT<real_t> &phi3a,
        Grid_FFT<real_t> &phi3b,
        std::array<Grid_FFT<real_t> *, 3> &A3);
}
