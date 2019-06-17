#pragma once

#include <array>

#include <cosmology_parameters.hh>
#include <transfer_function_plugin.hh>
#include <logger.hh>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>

/*!
 * @class CosmologyCalculator
 * @brief provides functions to compute cosmological quantities
 *
 * This class provides member functions to compute cosmological quantities
 * related to the Friedmann equations and linear perturbation theory
 */
class CosmologyCalculator
{
private:
    static constexpr double REL_PRECISION = 1e-5;

    real_t integrate(double (*func)(double x, void *params), double a, double b, void *params) const
    {
        gsl_function F;
        F.function = func;
        F.params = params;

        double result;
        double error;

        gsl_set_error_handler_off();
        gsl_integration_workspace *w = gsl_integration_workspace_alloc(100000);
        gsl_integration_qag(&F, a, b, 0, REL_PRECISION, 100000, 6, w, &result, &error);

        gsl_integration_workspace_free(w);

        gsl_set_error_handler(NULL);

        if (error / result > REL_PRECISION)
            csoca::wlog << "no convergence in function 'integrate', rel. error=" << error / result << std::endl;

        return (real_t)result;
    }

public:
    //! data structure to store cosmological parameters
    CosmologyParameters cosmo_param_;

    //! pointer to an instance of a transfer function plugin
    //TransferFunction_plugin *ptransfer_fun_;
    std::unique_ptr<TransferFunction_plugin> transfer_function_;


    //! constructor for a cosmology calculator object
    /*!
	 * @param acosmo a cosmological parameters structure
	 * @param pTransferFunction pointer to an instance of a transfer function object
	 */

    explicit CosmologyCalculator(ConfigFile &cf)
    : cosmo_param_(cf)
    {   
        csoca::ilog << "-----------------------------------------------------------------------------" << std::endl;
        transfer_function_ = std::move(select_TransferFunction_plugin(cf));
        transfer_function_->intialise();
        cosmo_param_.pnorm = this->ComputePNorm();
        cosmo_param_.sqrtpnorm = std::sqrt(cosmo_param_.pnorm);
        csoca::ilog << std::setw(40) << std::left << "TF supports distinct CDM+baryons" << " : " << (transfer_function_->tf_is_distinct()? "yes" : "no") << std::endl;
        csoca::ilog << std::setw(40) << std::left << "TF maximum wave number" << " : " << transfer_function_->get_kmax() << " h/Mpc" << std::endl;
    }

    //! Write out a correctly scaled power spectrum at time a
    void WritePowerspectrum( real_t a, std::string fname ) const
    {
        const real_t Dplus0 = this->CalcGrowthFactor(a) / this->CalcGrowthFactor(1.0);

        if( CONFIG::MPI_task_rank==0 )
        {
            double kmin = std::max(1e-4,transfer_function_->get_kmin());

            // write power spectrum to a file
            std::ofstream ofs(fname.c_str());
            std::stringstream ss; ss << " (a=" << a <<")";
            ofs << "# " << std::setw(18) << "k [h/Mpc]"
                        << std::setw(20) << ("P_dtot(k)"+ss.str()) 
                        << std::setw(20) << ("P_dcdm(k)"+ss.str())
                        << std::setw(20) << ("P_dbar(k)"+ss.str())
                        << std::setw(20) << ("P_dtot(K) (a=1)")
                        << std::setw(20) << ("P_tcdm(k)"+ss.str()) 
                        << std::setw(20) << ("P_tbar(k)"+ss.str())
                        << std::endl;
            for( double k=kmin; k<transfer_function_->get_kmax(); k*=1.05 ){
                ofs << std::setw(20) << std::setprecision(10) << k
                    << std::setw(20) << std::setprecision(10) << std::pow(this->GetAmplitude(k, total) * Dplus0, 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->GetAmplitude(k, cdm) * Dplus0, 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->GetAmplitude(k, baryon) * Dplus0, 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->GetAmplitude(k, total), 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->GetAmplitude(k, vcdm) * Dplus0, 2.0)
                    << std::setw(20) << std::setprecision(10) << std::pow(this->GetAmplitude(k, vbaryon) * Dplus0, 2.0)
                    << std::endl;
            }
        }

        csoca::ilog << "Wrote power spectrum at a=" << a << " to file \'" << fname << "\'" << std::endl;
    }


    const CosmologyParameters &GetParams(void) const
    {
        return cosmo_param_;
    }

    //! returns the amplitude of amplitude of the power spectrum
    /*!
	 * @param k the wave number in h/Mpc
	 * @param a the expansion factor of the universe
	 * @returns power spectrum amplitude for wave number k at time a
	 */
    inline real_t Power(real_t k, real_t a)
    {
        real_t Dplus = CalcGrowthFactor(a);
        real_t DplusOne = CalcGrowthFactor(1.0);
        real_t pNorm = ComputePNorm();
        Dplus /= DplusOne;
        DplusOne = 1.0;
        real_t scale = Dplus / DplusOne;
        return pNorm * scale * scale * TransferSq(k) * pow((double)k, (double)cosmo_param_.nspect);
    }

    inline static double H_of_a(double a, void *Params)
    {
        CosmologyParameters *cosm = (CosmologyParameters *)Params;
        double a2 = a * a;
        double Ha = sqrt(cosm->Omega_m / (a2 * a) + cosm->Omega_k / a2 + cosm->Omega_DE * pow(a, -3. * (1. + cosm->w_0 + cosm->w_a)) * exp(-3. * (1.0 - a) * cosm->w_a));
        return Ha;
    }

    inline static double Hprime_of_a(double a, void *Params) 
    {
        CosmologyParameters *cosm = (CosmologyParameters *)Params;
        double a2 = a * a;
        double H = H_of_a(a, Params);
        double Hprime = 1 / (a * H) * (-1.5 * cosm->Omega_m / (a2 * a) - cosm->Omega_k / a2 - 1.5 * cosm->Omega_DE * pow(a, -3. * (1. + cosm->w_0 + cosm->w_a)) * exp(-3. * (1.0 - a) * cosm->w_a) * (1. + cosm->w_0 + (1. - a) * cosm->w_a));
        return Hprime;
    }

    //! Integrand used by function CalcGrowthFactor to determine the linear growth factor D+
    inline static double GrowthIntegrand(double a, void *Params) 
    {
        double Ha = a * H_of_a(a, Params);
        return 2.5 / (Ha * Ha * Ha);
    }

    //! Computes the linear theory growth factor D+
    /*! Function integrates over member function GrowthIntegrand and computes
    *                      /a
    *   D+(a) = 5/2 H(a) * |  [a'^3 * H(a')^3]^(-1) da'
    *                      /0
    */
    real_t CalcGrowthFactor(real_t a) const
    {
        real_t integral = integrate(&GrowthIntegrand, 0.0, a, (void *)&cosmo_param_);
        return H_of_a(a, (void *)&cosmo_param_) * integral;
    }

    //! Compute the factor relating particle displacement and velocity
    /*! Function computes
    *
    *  vfac = a^2 * H(a) * dlogD+ / d log a = a^2 * H'(a) + 5/2 * [ a * D+(a) * H(a) ]^(-1)
    *
    */
    real_t CalcVFact(real_t a) const
    {
        real_t Dp = CalcGrowthFactor(a);
        real_t H = H_of_a(a, (void *)&cosmo_param_);
        real_t Hp = Hprime_of_a(a, (void *)&cosmo_param_);
        real_t a2 = a * a;

        return (a2 * Hp + 2.5 / (a * Dp * H)) * 100.0;
    }

    //! Integrand for the sigma_8 normalization of the power spectrum
    /*! Returns the value of the primordial power spectrum multiplied with 
     the transfer function and the window function of 8 Mpc/h at wave number k */
    static double dSigma8(double k, void *pParams)
    {
        if (k <= 0.0)
            return 0.0f;

        CosmologyCalculator *pcc = reinterpret_cast<CosmologyCalculator*>(pParams);
        
        double x = k * 8.0;
        double w = 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
        static double nspect = (double)pcc->cosmo_param_.nspect;
        double tf = pcc->transfer_function_->compute(k, total);

        //... no growth factor since we compute at z=0 and normalize so that D+(z=0)=1
        return k * k * w * w * pow((double)k, (double)nspect) * tf * tf;
    }

    //! Integrand for the sigma_8 normalization of the power spectrum
    /*! Returns the value of the primordial power spectrum multiplied with 
	 the transfer function and the window function of 8 Mpc/h at wave number k */
    static double dSigma8_0(double k, void *pParams)
    {
        if (k <= 0.0)
            return 0.0f;

        CosmologyCalculator *pcc = reinterpret_cast<CosmologyCalculator*>(pParams);
       
        double x = k * 8.0;
        double w = 3.0 * (sin(x) - x * cos(x)) / (x * x * x);
        static double nspect = (double)pcc->cosmo_param_.nspect;
        double tf = pcc->transfer_function_->compute(k, total0);

        //... no growth factor since we compute at z=0 and normalize so that D+(z=0)=1
        return k * k * w * w * pow((double)k, (double)nspect) * tf * tf;
    }

    //! Computes the square of the transfer function
    /*! Function evaluates the supplied transfer function ptransfer_fun_
	 * and returns the square of its value at wave number k
	 * @param k wave number at which to evaluate the transfer function
	 */
    inline real_t TransferSq(real_t k) const
    {
        //.. parameter supplied transfer function
        real_t tf1 = transfer_function_->compute(k, total);
        return tf1 * tf1;
    }

    //! Computes the amplitude of a mode from the power spectrum
    /*! Function evaluates the supplied transfer function ptransfer_fun_
	 * and returns the amplitude of fluctuations at wave number k at z=0
	 * @param k wave number at which to evaluate
	 */
    inline real_t GetAmplitude(real_t k, tf_type type) const
    {
        return std::pow(k, 0.5 * cosmo_param_.nspect) * transfer_function_->compute(k, type) * cosmo_param_.sqrtpnorm;
    }

    //! Computes the normalization for the power spectrum
    /*!
	 * integrates the power spectrum to fix the normalization to that given
	 * by the sigma_8 parameter
	 */
    real_t ComputePNorm(void)
    {
        real_t sigma0, kmin, kmax;
        kmax = transfer_function_->get_kmax();
        kmin = transfer_function_->get_kmin();

        if (!transfer_function_->tf_has_total0())
            sigma0 = 4.0 * M_PI * integrate(&dSigma8, (double)kmin, (double)kmax, this );
        else
            sigma0 = 4.0 * M_PI * integrate(&dSigma8_0, (double)kmin, (double)kmax, this );

        return cosmo_param_.sigma8 * cosmo_param_.sigma8 / sigma0;
    }
};

//! compute the jeans sound speed
/*! given a density in g/cm^-3 and a mass in g it gives back the sound
 *  speed in cm/s for which the input mass is equal to the jeans mass
 *  @param rho density 
 *  @param mass mass scale
 *  @returns jeans sound speed
 */
inline double jeans_sound_speed(double rho, double mass)
{
    const double G = 6.67e-8;
    return pow(6.0 * mass / M_PI * sqrt(rho) * pow(G, 1.5), 1.0 / 3.0);
}