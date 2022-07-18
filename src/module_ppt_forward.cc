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

#include <general.hh>
#include <grid_fft.hh>
#include <operators.hh>
#include <convolution.hh>
#include <testing.hh>

#include <module_ppt_forward.hh>
#include <particle_generator.hh>
#include <particle_plt.hh>

#include <unistd.h> // for unlink


/**
 * @brief the namespace encapsulating the main IC generation routines
 * 
 */
namespace ppt_forward_model{

//! global RNG object
std::unique_ptr<RNG_plugin> the_random_number_generator;

//! global output object
std::unique_ptr<output_plugin> the_output_plugin;

//! global cosmology object (calculates all things cosmological)
std::unique_ptr<cosmology::calculator>  the_cosmo_calc;

/**
 * @brief Initialises all global objects
 * 
 * @param the_config reference to config_file object
 * @return int 0 if successful
 */
int initialise( config_file& the_config )
{
    the_random_number_generator = std::move(select_RNG_plugin(the_config));
    the_cosmo_calc              = std::make_unique<cosmology::calculator>(the_config);
    the_output_plugin           = std::move(select_output_plugin(the_config, the_cosmo_calc));
    
    return 0;
}

/**
 * @brief Reset all global objects
 * 
 */
void reset () {
    the_random_number_generator.reset();
    the_output_plugin.reset();
    the_cosmo_calc.reset();
}


/**
 * @brief Main driver routine for IC generation, everything interesting happens here
 * 
 * @param the_config reference to the config_file object
 * @return int 0 if successful
 */
int run( config_file& the_config )
{
    //--------------------------------------------------------------------------------------------------------
    // Read run parameters
    //--------------------------------------------------------------------------------------------------------

    //--------------------------------------------------------------------------------------------------------
    //! number of resolution elements per dimension
    const size_t ngrid = the_config.get_value<size_t>("setup", "GridRes");

    //--------------------------------------------------------------------------------------------------------
    //! box side length in h-1 Mpc
    const real_t boxlen = the_config.get_value<double>("setup", "BoxLength");

    //--------------------------------------------------------------------------------------------------------
    //! starting redshift
    const real_t zstart = the_config.get_value<double>("setup", "zstart");

    //--------------------------------------------------------------------------------------------------------
    //! order of the LPT approximation 
    const int LPTorder = the_config.get_value_safe<double>("setup","LPTorder",100);

    //--------------------------------------------------------------------------------------------------------
    //! apply fixing of the complex mode amplitude following Angulo & Pontzen (2016) [https://arxiv.org/abs/1603.05253]
    const bool bDoFixing    = the_config.get_value_safe<bool>("setup", "DoFixing", false);
    const bool bDoInversion = the_config.get_value_safe<bool>("setup", "DoInversion", false);
    
    double Omega0 = the_cosmo_calc->cosmo_param_["Omega_m"];
    
    //--------------------------------------------------------------------------------------------------------
    //! do constrained ICs?
    const bool bAddConstrainedModes =  the_config.contains_key("random", "ConstraintFieldFile" );

    //--------------------------------------------------------------------------------------------------------

    const real_t astart = 1.0/(1.0+zstart);
    const real_t volfac(std::pow(boxlen / ngrid / 2.0 / M_PI, 1.5));

    the_cosmo_calc->write_powerspectrum(astart, "input_powerspec.txt" );
    the_cosmo_calc->write_transfer("input_transfer.txt" );

    // the_cosmo_calc->compute_sigma_bc();
    // abort();

    //--------------------------------------------------------------------
    // Compute LPT time coefficients
    //--------------------------------------------------------------------
    const real_t Dplus0 = the_cosmo_calc->get_growth_factor(astart);
    const real_t vfac   = the_cosmo_calc->get_vfact(astart);

    const real_t g1  = -Dplus0;
    const real_t g2  = ((LPTorder>1)? -3.0/7.0*Dplus0*Dplus0 : 0.0);
    const real_t g3  = ((LPTorder>2)? 1.0/3.0*Dplus0*Dplus0*Dplus0 : 0.0);
    const real_t g3c = ((LPTorder>2)? 1.0/7.0*Dplus0*Dplus0*Dplus0 : 0.0);

    // vfac = d log D+ / dt 
    // d(D+^2)/dt = 2*D+ * d D+/dt = 2 * D+^2 * vfac
    // d(D+^3)/dt = 3*D+^2* d D+/dt = 3 * D+^3 * vfac
    const real_t vfac1 =  vfac;
    const real_t vfac2 =  2*vfac;
    const real_t vfac3 =  3*vfac;

    //--------------------------------------------------------------------
    // Create arrays
    //--------------------------------------------------------------------

    // white noise field 
    Grid_FFT<real_t> wnoise({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    //... Fill the wnoise grid with a Gaussian white noise field, we do this first since the RNG might need extra memory
    music::ilog << "-------------------------------------------------------------------------------" << std::endl;
    music::ilog << "Generating white noise field...." << std::endl;

    the_random_number_generator->Fill_Grid(wnoise);
    
    wnoise.FourierTransformForward();

    //... Next, declare LPT related arrays, allocated only as needed by order
    Grid_FFT<real_t> phi({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> phi2({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen}, false); // do not allocate these unless needed
    
    // temporary storage of additional data
    Grid_FFT<real_t> tmp({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    //--------------------------------------------------------------------
    // Apply Normalisation factor and Angulo&Pontzen fixing or not
    //--------------------------------------------------------------------

    wnoise.apply_function_k( [&](auto wn){
        if (bDoFixing){
            wn = (std::fabs(wn) != 0.0) ? wn / std::fabs(wn) : wn;
        }
        return ((bDoInversion)? real_t{-1.0} : real_t{1.0}) * wn / volfac;
    });


    //--------------------------------------------------------------------
    // Compute the LPT terms....
    //--------------------------------------------------------------------

    //--------------------------------------------------------------------
    // Create convolution class instance for non-linear terms
    //--------------------------------------------------------------------
#if defined(USE_CONVOLVER_ORSZAG)
    OrszagConvolver<real_t> Conv({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
#elif defined(USE_CONVOLVER_NAIVE)
    NaiveConvolver<real_t> Conv({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
#endif
    //--------------------------------------------------------------------

    //======================================================================
    //... compute 1LPT displacement potential ....
    //======================================================================
    // phi = - delta / k^2

    music::ilog << "-------------------------------------------------------------------------------" << std::endl;
    music::ilog << "Generating LPT fields...." << std::endl;

    double wtime = get_wtime();
    music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(1) term" << std::flush;

    phi.FourierTransformForward(false);
    phi.assign_function_of_grids_kdep([&](auto k, auto wn) {
        real_t kmod = k.norm();
        ccomplex_t delta = wn * the_cosmo_calc->get_amplitude(kmod, delta_matter);
        return -delta / (kmod * kmod);
    }, wnoise);

    phi.zero_DC_mode();

    music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;

    //======================================================================
    //... compute 2LPT displacement potential ....
    //======================================================================
    if (LPTorder > 1)
    {
        phi2.allocate();
        phi2.FourierTransformForward(false);
        
        wtime = get_wtime();
        music::ilog << std::setw(40) << std::setfill('.') << std::left << "Computing phi(2) term" << std::flush;
        Conv.convolve_SumOfHessians(phi, {0, 0}, phi, {1, 1}, {2, 2}, op::assign_to(phi2));
        Conv.convolve_Hessians(phi, {1, 1}, phi, {2, 2}, op::add_to(phi2));
        Conv.convolve_Hessians(phi, {0, 1}, phi, {0, 1}, op::subtract_from(phi2));
        Conv.convolve_Hessians(phi, {0, 2}, phi, {0, 2}, op::subtract_from(phi2));
        Conv.convolve_Hessians(phi, {1, 2}, phi, {1, 2}, op::subtract_from(phi2));

        phi2.apply_InverseLaplacian();
        music::ilog << std::setw(20) << std::setfill(' ') << std::right << "took " << get_wtime() - wtime << "s" << std::endl;
    }

    ///... scale all potentials with respective growth factors
    phi *= g1;

    if (LPTorder > 1)
    {
        phi2 *= g2;
    }

    music::ilog << "-------------------------------------------------------------------------------" << std::endl;


    //==============================================================//
    // main output loop, loop over all species that are enabled
    //==============================================================//


    //======================================================================
    // use QPT to get density and velocity fields
    //======================================================================
    Grid_FFT<ccomplex_t> psi({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    Grid_FFT<real_t> rho({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});

    //======================================================================
    // initialise rho
    //======================================================================
    wnoise.FourierTransformForward();
    rho.FourierTransformForward(false);
    rho.assign_function_of_grids_kdep( [&]( auto k, auto wn ){
        return wn * the_cosmo_calc->get_amplitude(k.norm(), delta_matter);
    }, wnoise );
    rho.zero_DC_mode();
    rho.FourierTransformBackward();

    const double b1 = -0.0;
    
    rho.apply_function_r( [&]( auto prho ){
        return std::sqrt( 1.0 + std::max(b1 * prho, 0.0) );
    });

    //======================================================================
    // initialise psi = exp(i Phi(1)/hbar)
    //======================================================================
    phi.FourierTransformBackward();

    real_t maxdphi = -1.0;

    #pragma omp parallel for reduction(max:maxdphi)
    for( size_t i=0; i<phi.size(0)-1; ++i ){
        size_t ir = (i+1)%phi.size(0);
        for( size_t j=0; j<phi.size(1); ++j ){
            size_t jr = (j+1)%phi.size(1);    
            for( size_t k=0; k<phi.size(2); ++k ){
                size_t kr = (k+1)%phi.size(2);
                auto phic = phi.relem(i,j,k);

                auto dphixr = std::fabs(phi.relem(ir,j,k) - phic);
                auto dphiyr = std::fabs(phi.relem(i,jr,k) - phic);
                auto dphizr = std::fabs(phi.relem(i,j,kr) - phic);
                
                maxdphi = std::max(maxdphi,std::max(dphixr,std::max(dphiyr,dphizr)));
            }
        }
    }
    #if defined(USE_MPI)
        real_t local_maxdphi = maxdphi;
        MPI_Allreduce( &local_maxdphi, &maxdphi, 1, MPI::get_datatype<real_t>(), MPI_MAX, MPI_COMM_WORLD );
    #endif
    const real_t hbar_safefac = 1.01;
    const real_t hbar = maxdphi / M_PI / Dplus0 * hbar_safefac;
    music::ilog << "Semiclassical PT : hbar = " << hbar << " (limited by initial potential, safety=" << hbar_safefac << ")." << std::endl;
    
    if( LPTorder == 1 ){
        psi.assign_function_of_grids_r([hbar,Dplus0]( real_t pphi, real_t prho ){
            return prho * std::exp(ccomplex_t(0.0,1.0/hbar) * (pphi / Dplus0)); // divide by Dplus since phi already contains it
        }, phi, rho );
    }else if( LPTorder >= 2 ){
        phi2.FourierTransformBackward();
        // we don't have a 1/2 in the Veff term because pre-factor is already 3/7
        psi.assign_function_of_grids_r([hbar,Dplus0]( real_t pphi, real_t pphi2, real_t prho ){
            return prho * std::exp(ccomplex_t(0.0,1.0/hbar) * (pphi + pphi2) / Dplus0);
        }, phi, phi2, rho );
    }

    //======================================================================
    // evolve wave-function (one drift step) psi = psi *exp(-i hbar *k^2 dt / 2)
    //======================================================================
    psi.FourierTransformForward();
    psi.apply_function_k_dep([hbar,Dplus0]( auto epsi, auto k ){
        auto k2 = k.norm_squared();
        return epsi * std::exp( - ccomplex_t(0.0,0.5)*hbar* k2 * Dplus0);
    });
    psi.FourierTransformBackward();

    if( LPTorder >= 2 ){
        psi.assign_function_of_grids_r([&](auto ppsi, auto pphi2) {
            return ppsi * std::exp(ccomplex_t(0.0,1.0/hbar) * (pphi2) / Dplus0);
        }, psi, phi2);
    }

    //======================================================================
    // compute rho
    //======================================================================
    rho.assign_function_of_grids_r([&]( auto p ){
        auto pp = std::real(p)*std::real(p) + std::imag(p)*std::imag(p) - 1.0;
        return pp;
    }, psi);

    the_output_plugin->write_grid_data( rho, cosmo_species::dm, fluid_component::density );
    rho.Write_PowerSpectrum("input_powerspec_sampled_evolved_semiclassical.txt");
    rho.FourierTransformBackward();
    
    //======================================================================
    // compute  v
    //======================================================================
    // Grid_FFT<ccomplex_t> grad_psi({ngrid, ngrid, ngrid}, {boxlen, boxlen, boxlen});
    // const real_t vunit = Dplus0 * vfac / boxlen * the_output_plugin->velocity_unit();
    // for( int idim=0; idim<3; ++idim )
    // {
    //     grad_psi.FourierTransformBackward(false);
    //     grad_psi.copy_from(psi);
    //     grad_psi.FourierTransformForward();
    //     grad_psi.apply_function_k_dep([&](auto x, auto k) {
    //         return x * ccomplex_t(0.0,k[idim]);
    //     });
    //     grad_psi.FourierTransformBackward();
        
    //     tmp.FourierTransformBackward(false);
    //     tmp.assign_function_of_grids_r([&](auto ppsi, auto pgrad_psi, auto prho) {
    //             return vunit * std::real((std::conj(ppsi) * pgrad_psi - ppsi * std::conj(pgrad_psi)) / ccomplex_t(0.0, 2.0 / hbar)/real_t(1.0+prho));
    //         }, psi, grad_psi, rho);

    //     fluid_component fc = (idim==0)? fluid_component::vx : ((idim==1)? fluid_component::vy : fluid_component::vz );
    //     the_output_plugin->write_grid_data( tmp, cosmo_species::dm, fc );
    // }
        
    music::ilog << "-------------------------------------------------------------------------------" << std::endl;
        
    
    return 0;
}


} // end namespace ic_generator

