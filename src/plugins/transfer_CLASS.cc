//  transfer_CLASS.cc - This file is part of MUSIC -
//  a code to generate multi-scale initial conditions for cosmological simulations

//  Copyright (C) 2019  Oliver Hahn

#ifdef USE_CLASS

#include <cmath>
#include <string>
#include <vector>
#include <memory>
#include <array>
#include <ClassEngine.hh>

#include <general.hh>
#include <config_file.hh>
#include <transfer_function_plugin.hh>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_spline.h>

struct fluid_params{
    double H0, Omega_m, fb, fc;
};

// constexpr int nvar{4};

// define some operators on arrays
using vec4 = std::array<double,4>;
vec4 operator*( double x, vec4 y ){ return {x*y[0],x*y[1],x*y[2],x*y[3]}; }
vec4 operator/( vec4 y, double x ){ return {y[0]/x,y[1]/x,y[2]/x,y[3]/x}; }
vec4 operator+( vec4 x, vec4 y ){ return {x[0]+y[0],x[1]+y[1],x[2]+y[2],x[3]+y[3]}; }
vec4 operator-( vec4 x, vec4 y ){ return {x[0]-y[0],x[1]-y[1],x[2]-y[2],x[3]-y[3]}; }

void backscale_twofluid( double astart, double aend, vec4& instate, fluid_params& p, vec4& outstate );

class transfer_CLASS_plugin : public TransferFunction_plugin {

private:
    std::vector<double> tab_lnk_, tab_dtot_, tab_dc_, tab_db_, tab_ttot_, tab_tc_, tab_tb_;
    gsl_interp_accel *gsl_ia_dtot_, *gsl_ia_dc_, *gsl_ia_db_, *gsl_ia_ttot_, *gsl_ia_tc_, *gsl_ia_tb_;
    gsl_spline *gsl_sp_dtot_, *gsl_sp_dc_, *gsl_sp_db_, *gsl_sp_ttot_, *gsl_sp_tc_, *gsl_sp_tb_;
    double Omega_m_, Omega_b_, Omega_r_, N_ur_, zstart_, ztarget_, kmax_, kmin_, H0_, h_, fb_, fc_, gfac_;

    void ClassEngine_get_data( void ){
        std::vector<double> d_ncdm, t_ncdm, phi, psi;

        csoca::ilog << "Computing TF via ClassEngine..." << std::endl << " ztarget = " << ztarget_ << ", zstart = " << zstart_ << " ..." << std::flush;
        double wtime = get_wtime();
        
        ClassParams pars;
        pars.add("extra metric transfer functions", "yes");
        pars.add("z_pk",ztarget_);
        pars.add("P_k_max_h/Mpc", kmax_);
        pars.add("h",h_);
        pars.add("Omega_b",Omega_b_);
        pars.add("N_ur",N_ur_);
        pars.add("Omega_cdm",Omega_m_-Omega_b_);
        //pars.add("Omega_Lambda",1.0-Omega_m_);
        pars.add("Omega_k",0.0);
        pars.add("Omega_fld",0.0);
        pars.add("Omega_scf",0.0);
        pars.add("A_s",2.42e-9);
        pars.add("n_s",.961); // tnis doesn't matter for TF
        pars.add("output","dTk,vTk");
        pars.add("YHe",0.248);
        pars.add("lensing","no");
        pars.add("alpha_s",0.0);
        pars.add("P_k_ini type","analytic_Pk");
        pars.add("gauge","synchronous");
//         lensing = no
// ic = ad
// gauge = synchronous
// P_k_ini type = analytic_Pk
// k_pivot = 0.05
// A_s = 2.215e-9
// n_s = 0.96
// alpha_s = 0.

        pars.add("k_per_decade_for_pk",50);
        pars.add("k_per_decade_for_bao",50);
        pars.add("tol_perturb_integration",1e-8);
        pars.add("tol_background_integration",1e-9);
        pars.add("compute damping scale","yes");
        // pars.add("evolver",1);
        
        pars.add("z_reio",10.0); // make sure reionisation is not included

        std::unique_ptr<ClassEngine> CE = std::make_unique<ClassEngine>(pars, false);

        CE->getTk(ztarget_, tab_lnk_, tab_dc_, tab_db_, d_ncdm, tab_dtot_,
                tab_tc_, tab_tb_, t_ncdm, tab_ttot_, phi, psi );
        double astart = 1.0/(1.0+ztarget_);
        // assume dtot = fc * dc + fb * db -> dc = (dtot-fB * db)/fc
        for( size_t i=0; i<tab_lnk_.size(); ++i ){
        //   tab_dc_[i] = (tab_dtot_[i] - fb_ * tab_db_[i]) / fc_;

          std::cerr << tab_tb_[i] << " " << tab_tc_[i] << " " << -tab_dtot_[i]*0.514704 << std::endl;
          // tab_tc_[i] = -0.514704 * tab_dtot_[i];
          // tab_tb_[i] = -0.514704 * tab_dtot_[i];
          
        }

        wtime = get_wtime() - wtime;
        csoca::ilog << "   took " << wtime << " s / " << tab_lnk_.size() << " modes."  << std::endl;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    vec4 k1_, k2_, k3_, k4_; //!< RK4 temporary states

    vec4 RK4_step( const vec4& y, double t, double dt ){
        k1_ = dt * f( y, t);
        k2_ = dt * f( y+k1_/2, t+dt/2 );
        k3_ = dt * f( y+k2_/2, t+dt/2 );
        k4_ = dt * f( y+k3_, t+dt );

        return y + (k1_ + 2*k2_ + 2*k3_ + k4_ )/6;
    }

    double Hubble_a( double a ) const {
      return H0_ * std::sqrt(Omega_r_/(a*a*a*a)+Omega_m_/(a*a*a)+(1.0-Omega_m_-Omega_r_));
    }

    vec4 f( const vec4 y, const double a ) const
    {
        auto deltac = y[0];
        auto thetac = y[1]; 
        auto deltab = y[2];
        auto thetab = y[3]; 
        double adot = Hubble_a(a)*a;

        vec4 ff;
        ff[0] = -thetac/(a*adot); 
        ff[1] = -thetac/a - gfac_ * (fc_*deltac+fb_*deltab)/(a*a*adot);
        ff[2] = -thetab/(a*adot); 
        ff[3] = -thetab/a - gfac_ * (fc_*deltac+fb_*deltab)/(a*a*adot);
        // ff[0] = -thetac/(a*adot); 
        // ff[1] = -thetac/a - gfac_ * (deltac)/(a*a*adot);
        // ff[2] = -thetab/(a*adot); 
        // ff[3] = -thetab/a - gfac_ * (deltac)/(a*a*adot);
        return ff;
    }

    void BackScaleTF( void ){
      csoca::ilog << "Back-Scaling two-fluid TF from z=" << ztarget_ << " to z=" << zstart_ << "..." << std::endl; 
      std::ofstream ofs("backsc.txt");
      vec4 state, state0;
      for( size_t i=0; i<tab_lnk_.size(); ++i ){
        double a = 1.0/(1.0+ztarget_);
        double afinal = 1.0/(1.0+zstart_);
        double da = 1e-3;//afinal/100.0;

        state0[0] = -tab_dc_[i];
        state0[1] = -tab_tc_[i]*Hubble_a(a);
        state0[2] = -tab_db_[i];
        state0[3] = -tab_tb_[i]*Hubble_a(a);
        state = state0;
        int istep{0};
        if( zstart_ > ztarget_ ){
          while( a > afinal ){
            state = RK4_step(state, a, -da*a);
            a-=da*a;
            da = std::min(da*a,a-afinal)/a;
            ++istep;
          }
        }else{
          while( a < afinal ){
            state = RK4_step(state, a, da*a);
            a+=da*a;
            da = std::min(da*a,afinal-a)/a;
            ++istep;
          }
        }
        // std::cerr << "istep=" << istep << std::endl;
        ofs << std::exp(tab_lnk_[i]) << " ";
        for( int j=0; j<4; ++j )
          ofs << state0[j] << " " << state[j] << " ";
        ofs << std::endl;
      }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////////////////////////////////////////////////////////////////
    

    // void BackScaleTF( void ){
    //   vec4 state, endstate;
    //   fluid_params p;
    //   p.H0 = 100.0*h_;
    //   p.Omega_m = Omega_m_;
    //   p.fb = Omega_b_/Omega_m_;
    //   p.fc = 1.0-p.fb;

    //   csoca::ilog << "Back-Scaling two-fluid TF from z=" << ztarget_ << " to z=" << zstart_ << "..." << std::endl; 
    //   std::ofstream ofs("backsc.txt");
    //   for( size_t i=0; i<tab_lnk_.size(); ++i ){
    //     state[0] = -tab_dc_[i];
    //     state[1] = -1e6 * tab_tc_[i];
    //     state[2] = -tab_db_[i];
    //     state[3] = -tab_tb_[i];

    //     backscale_twofluid( 1.0/(1.0+ztarget_), 1.0/(1.0+zstart_), state, p, endstate );

    //     ofs << std::exp(tab_lnk_[i]) << " " << state[0] << " " << endstate[0] << " " << state[2] << " " << endstate[2] << std::endl;
    //   }

    // }

public:
  explicit transfer_CLASS_plugin( ConfigFile &cf)
  : TransferFunction_plugin(cf)
  { 
    h_       = pcf_->GetValue<double>("cosmology","H0") / 100.0; 
    Omega_m_ = pcf_->GetValue<double>("cosmology","Omega_m"); 
    Omega_b_ = pcf_->GetValue<double>("cosmology","Omega_b");
    N_ur_    = pcf_->GetValueSafe<double>("cosmology","N_ur", 3.046);
    ztarget_ = pcf_->GetValueSafe<double>("setup","ztarget",0.0);
    zstart_  = pcf_->GetValue<double>("setup","zstart");
    double lbox = pcf_->GetValue<double>("setup","BoxLength");
    int nres = pcf_->GetValue<double>("setup","GridRes");

    double Tcmb = 2.726;
    double Omega_g = 4.48146636e-7 * std::pow(Tcmb,4) /h_/h_;
    double Omega_nu = N_ur_ * (7./8) * std::pow(4./11,4./3) * Omega_g;

    Omega_r_ = 0.0; //Omega_g+Omega_nu;
    std::cerr << "Omega_r_ = " << Omega_r_ << std::endl;
    kmax_    = 2.0*M_PI/lbox * nres/2 * std::sqrt(3.0) * 2.0; // 200% of spatial diagonal
    H0_ = 100.0*h_;
    fb_ = Omega_b_/Omega_m_;
    fc_ = 1.0-Omega_b_/Omega_m_;
    gfac_ = 1.5*H0_*H0_*Omega_m_;
    
    this->ClassEngine_get_data();

    this->BackScaleTF();
    
    gsl_ia_dtot_ = gsl_interp_accel_alloc();
    gsl_ia_dc_   = gsl_interp_accel_alloc();
    gsl_ia_db_   = gsl_interp_accel_alloc();
    gsl_ia_ttot_ = gsl_interp_accel_alloc();
    gsl_ia_tc_   = gsl_interp_accel_alloc();
    gsl_ia_tb_   = gsl_interp_accel_alloc();

    gsl_sp_dtot_ = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_dc_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_db_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_ttot_ = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_tc_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());
    gsl_sp_tb_   = gsl_spline_alloc(gsl_interp_cspline, tab_lnk_.size());

    gsl_spline_init(gsl_sp_dtot_, &tab_lnk_[0], &tab_dtot_[0], tab_lnk_.size());
    gsl_spline_init(gsl_sp_dc_,   &tab_lnk_[0], &tab_dc_[0],   tab_lnk_.size());
    gsl_spline_init(gsl_sp_db_,   &tab_lnk_[0], &tab_db_[0],   tab_lnk_.size());
    gsl_spline_init(gsl_sp_ttot_, &tab_lnk_[0], &tab_ttot_[0], tab_lnk_.size());
    gsl_spline_init(gsl_sp_tc_,   &tab_lnk_[0], &tab_tc_[0],   tab_lnk_.size());
    gsl_spline_init(gsl_sp_tb_,   &tab_lnk_[0], &tab_tb_[0],   tab_lnk_.size());

    kmin_ = std::exp(tab_lnk_[0]);
  
    tf_distinct_ = true; 
    tf_withvel_  = true; 
  }
    
  ~transfer_CLASS_plugin(){
    gsl_spline_free(gsl_sp_dtot_);
    gsl_spline_free(gsl_sp_dc_);
    gsl_spline_free(gsl_sp_db_);
    gsl_spline_free(gsl_sp_ttot_);
    gsl_spline_free(gsl_sp_tc_);
    gsl_spline_free(gsl_sp_tb_);

    gsl_interp_accel_free(gsl_ia_dtot_);
    gsl_interp_accel_free(gsl_ia_dc_);
    gsl_interp_accel_free(gsl_ia_db_);
    gsl_interp_accel_free(gsl_ia_ttot_);
    gsl_interp_accel_free(gsl_ia_tc_);
    gsl_interp_accel_free(gsl_ia_tb_);
  }

  inline double compute(double k, tf_type type) const {
      gsl_spline *splineT = nullptr;
      gsl_interp_accel *accT = nullptr;
      switch(type){
          case total:   splineT = gsl_sp_dtot_; accT = gsl_ia_dtot_; break;
          case cdm:     splineT = gsl_sp_dc_;   accT = gsl_ia_dc_;   break;
          case baryon:  splineT = gsl_sp_db_;   accT = gsl_ia_db_;   break;
          case vtotal:  splineT = gsl_sp_ttot_; accT = gsl_ia_ttot_; break;
          case vcdm:    splineT = gsl_sp_tc_;   accT = gsl_ia_tc_;   break;
          case vbaryon: splineT = gsl_sp_tb_;   accT = gsl_ia_tb_;   break;
          default:
            throw std::runtime_error("Invalid type requested in transfer function evaluation");
      }

      double d = (k<=kmin_)? gsl_spline_eval(splineT, std::log(kmin_), accT) 
        : gsl_spline_eval(splineT, std::log(k*h_), accT);
      return -d/(k*k);
  }

  inline double get_kmin(void) const { return std::exp(tab_lnk_[0])/h_; }
  inline double get_kmax(void) const { return std::exp(tab_lnk_[tab_lnk_.size()-1])/h_; }
};

namespace {
TransferFunction_plugin_creator_concrete<transfer_CLASS_plugin> creator("CLASS");
}


// #include <gsl/gsl_errno.h>
// #include <gsl/gsl_matrix.h>
// #include <gsl/gsl_odeiv2.h>




// int fluid_eq( double a, const double y[], double f[], void* params )
// {
//     fluid_params *p = reinterpret_cast<fluid_params*>(params);
//     double delta_c = y[0];
//     double theta_c = y[1];
//     double delta_b = y[2];
//     double theta_b = y[3];
//     // a=-a;

//     double adot = p->H0 * std::sqrt( p->Omega_m / a + (1.0-p->Omega_m) * a * a );
//     double Gfac = 1.5*p->H0*p->H0*p->Omega_m;

//     f[0] = (-theta_c / (a*adot));
//     f[1] = (-theta_c / a - Gfac * delta_c / (a*a*adot) );
//     // f[1] = (-theta_c / a - Gfac * delta_c / (a*a*adot) );
//     //f[1] = -(theta_c / a + Gfac * (p->fc * delta_c + p->fb * delta_b) / (a*a*adot));
//     //f[2] = -(theta_b / (a*adot));
//     //f[3] = -(theta_b / a + Gfac * (p->fc * delta_c + p->fb * delta_b) / (a*a*adot));

//     return GSL_SUCCESS;
// }

// void backscale_twofluid( double astart, double aend, vec4& instate, fluid_params& p, vec4& outstate )
// {
//     static const gsl_odeiv2_step_type * T = gsl_odeiv2_step_rk4;
//     const size_t nvar = instate.size();

//     gsl_odeiv2_step    *s = gsl_odeiv2_step_alloc (T, 4);
//     gsl_odeiv2_control *c = gsl_odeiv2_control_yp_new (1e-10,1e-10);
//     gsl_odeiv2_evolve  *e = gsl_odeiv2_evolve_alloc (4);

//     gsl_odeiv2_system sys = {fluid_eq, nullptr, nvar, reinterpret_cast<void*> (&p)};

//     double yk[nvar];
//     for( size_t ivar=0;ivar<nvar;++ivar ){
//         yk[ivar] = instate[ivar];
//     }

//     double step=1e-6;
//     double a = astart;
//     // aend = -aend;
//     while( a<aend ){
//         int status = gsl_odeiv2_evolve_apply (e, c, s, &sys, &a, aend, &step, &yk[0]);
//         // std::cerr << a << " " << step << std::endl;
//         if (status != GSL_SUCCESS)
//         {
//             std::cerr << "Error in gsl_odeiv_evolve_apply during iteration. Skipping mode\n";
//             continue;
//         }
//     }
//     // abort();

//     for( size_t ivar=0;ivar<nvar;++ivar ){
//         outstate[ivar] = yk[ivar];
//     }


//     gsl_odeiv2_evolve_free (e);
//     gsl_odeiv2_control_free (c);
//     gsl_odeiv2_step_free (s);
// }


#endif // USE_CLASS
