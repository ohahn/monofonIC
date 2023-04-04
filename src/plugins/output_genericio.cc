#ifdef ENABLE_GENERICIO

#include <output_plugin.hh>
#include <GenericIO.h>
#include <mpi.h>

class genericio_output_plugin : public output_plugin
{
protected:
    real_t lunit_, vunit_, munit_;
    bool hacc_hydro_;
    float hacc_etamax_;
    float hh_value_, rho_value_, mu_value_, uu_value_;
    std::vector<int64_t> ids;
    std::vector<float> xx, yy, zz;
    std::vector<float> vx, vy, vz;
    std::vector<float> mass, hh, uu;
    std::vector<float> mu, phi, rho;
    std::vector<float> zmet, yhe;
    std::vector<uint16_t> mask;

public:
    //! constructor
    explicit genericio_output_plugin(config_file &cf, std::unique_ptr<cosmology::calculator> &pcc)
            : output_plugin(cf, pcc, "GenericIO")
    {
        const double astart = 1.0 / (1.0 + cf_.get_value<double>("setup", "zstart"));
        const double hubble_param = pcc->cosmo_param_["h"];
        lunit_ = cf_.get_value<double>("setup", "BoxLength");
        vunit_ = lunit_/astart;
        hacc_hydro_ = cf_.get_value_safe<bool>("output", "GenericIO_HACCHydro", false);
        // initial smoothing length is mean particle seperation
        hh_value_ = lunit_ / cf_.get_value<float>("setup", "GridRes");

        double rhoc = 27.7519737 * 1e10; // in h^2 M_sol / Mpc^3
        munit_ = rhoc * std::pow(cf_.get_value<double>("setup", "BoxLength"), 3);

        if(hacc_hydro_) {
            const double omegab = pcc_->cosmo_param_["Omega_b"];
            const double gamma  = cf_.get_value_safe<double>("cosmology", "gamma", 5.0 / 3.0);
            const double YHe    = pcc_->cosmo_param_["YHe"];
            const double Tcmb0  = pcc_->cosmo_param_["Tcmb"];

            // compute gas internal energy
            const double npol = (fabs(1.0 - gamma) > 1e-7) ? 1.0 / (gamma - 1.) : 1.0;
            const double unitv = 1e5; // km/s to cm/s
            const double adec = 1.0 / (160. * std::pow(omegab * hubble_param * hubble_param / 0.022, 2.0 / 5.0));
            const double Tini = astart < adec ? Tcmb0 / astart : Tcmb0 / astart / astart * adec;
            const double mu = (Tini > 1.e4) ? 4.0 / (8. - 5. * YHe) : 4.0 / (1. + 3. * (1. - YHe));
            const double ceint = 1.3806e-16 / 1.6726e-24 * Tini * npol / mu / unitv / unitv;

            uu_value_ = ceint/astart/astart;  // probably needs a scale factor correction (might be in physical units).
            mu_value_ = mu;

            music::ilog.Print("HACC : calculated redshift of decoupling: z_dec = %.2f", 1./adec - 1.);
            music::ilog.Print("HACC : set initial gas temperature to %.2e K/mu", Tini / mu);
            music::ilog.Print("HACC : set initial internal energy to %.2e km^2/s^2", ceint);

            rho_value_ = omegab * rhoc;
        }
    }

    output_type write_species_as(const cosmo_species &) const { return output_type::particles; }

    real_t position_unit() const { return lunit_; }

    real_t velocity_unit() const { return vunit_; }

    real_t mass_unit() const { return munit_; }

    bool has_64bit_reals() const { return false; }

    bool has_64bit_ids() const { return true; }

    void write_particle_data(const particle::container &pc, const cosmo_species &s, double Omega_species)
    {
        double boxmass = Omega_species * munit_;
        double particle_mass = boxmass / pc.get_global_num_particles();

        size_t npart = pc.get_local_num_particles();
        xx.reserve(xx.size() + npart);
        yy.reserve(yy.size() + npart);
        zz.reserve(zz.size() + npart);
        vx.reserve(vx.size() + npart);
        vy.reserve(vy.size() + npart);
        vz.reserve(vz.size() + npart);
        ids.reserve(ids.size() + npart);
        mask.reserve(mask.size() + npart);
        // phi doesn't need to be initialized, just needs to be present in data
        phi.resize(phi.size() + npart, 0.0f);


        auto _pos = reinterpret_cast<const float*>(pc.get_pos32_ptr());
        auto _vel = reinterpret_cast<const float*>(pc.get_vel32_ptr());
        auto _ids = reinterpret_cast<const uint64_t*>(pc.get_ids64_ptr());
        auto _mass = reinterpret_cast<const float*>(pc.get_mass32_ptr());

        for(size_t i=0; i<npart; ++i) {
            xx.push_back(fmod(_pos[3*i + 0]+lunit_, lunit_));
            yy.push_back(fmod(_pos[3*i + 1]+lunit_, lunit_));
            zz.push_back(fmod(_pos[3*i + 2]+lunit_, lunit_));
            vx.push_back(_vel[3*i + 0]);
            vy.push_back(_vel[3*i + 1]);
            vz.push_back(_vel[3*i + 2]);
            mask.push_back(s == cosmo_species::baryon ? 1<<2 : 0);
        }

        std::copy(_ids, _ids+npart, std::back_inserter(ids));

        if(hacc_hydro_) {
            size_t prev_size = mass.size();
            size_t new_size = prev_size + npart;

            if(pc.bhas_individual_masses_) {
                std::copy(_mass, _mass+npart, std::back_inserter(mass));
            } else {
                mass.resize(new_size);
                std::fill(mass.begin() + prev_size, mass.end(), particle_mass);
            }

            hh.resize(new_size, hh_value_);
            uu.resize(new_size, s == cosmo_species::baryon ? uu_value_ : 0.0f);
            rho.resize(new_size, rho_value_);
            mu.resize(new_size, s == cosmo_species::baryon ? mu_value_ : 0.0f);
            zmet.resize(new_size, 0.0f);
            yhe.resize(new_size, 0.0f);
        }
    }

    ~genericio_output_plugin() override {
        gio::GenericIO writer(MPI_COMM_WORLD, fname_, gio::GenericIO::FileIO::FileIOMPI);
        writer.setPhysOrigin(0., -1);
        writer.setPhysScale(lunit_, -1);
        writer.setNumElems(xx.size());
        writer.addVariable("x", xx);
        writer.addVariable("y", yy);
        writer.addVariable("z", zz);
        writer.addVariable("vx", vx);
        writer.addVariable("vy", vy);
        writer.addVariable("vz", vz);
        writer.addVariable("id", ids);
        writer.addVariable("phi", phi);
        writer.addVariable("mask", mask);
        if(hacc_hydro_) {
            writer.addVariable("mass", mass);
            writer.addVariable("hh", hh);
            writer.addVariable("uu", uu);
            writer.addVariable("rho", rho);
            writer.addVariable("mu", mu);
            writer.addVariable("zmet", zmet);
            writer.addVariable("yhe", yhe);
        }
        writer.write();
        MPI_Barrier(MPI_COMM_WORLD);
    }
};

namespace
{
output_plugin_creator_concrete<genericio_output_plugin> creator1("genericio");
} // namespace

#endif // ENABLE_GENERICIO