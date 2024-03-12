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

#include <unistd.h> // for unlink
#include <sys/types.h>
#include <sys/stat.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

#include <general.hh>
#include <output_plugin.hh>

//! Implementation of class grafic2_output_plugin
/*!
 This class implements a grafic-2 (cf. Bertschinger 2001) compatible
 output format. With some RAMSES extras.
*/
class grafic2_output_plugin : public output_plugin
{
private:
    struct header
    {
        int n1, n2, n3;
        float dxini0;
        float xoff10, xoff20, xoff30;
        float astart0, omega_m0, omega_l0, h00;
    };

    std::string get_file_name(const cosmo_species &s, const fluid_component &c) const;
    void write_ramses_namelist(void) const;

protected:
    header header_;
    real_t lunit_, vunit_, munit_, omegab_;
    uint32_t levelmin_;
    bool bhavebaryons_;
    std::vector<float> data_buf_, data_buf_write_;
    std::string dirname_;
    bool bUseSPT_;

public:
    //! constructor
    explicit grafic2_output_plugin(config_file &cf, std::unique_ptr<cosmology::calculator> &pcc)
        : output_plugin(cf, pcc, "GRAFIC2/RAMSES")
    {
        lunit_ = 1.0;
        vunit_ = 1.0;

        double
            boxlength = cf_.get_value<double>("setup", "BoxLength"),
            zstart = cf_.get_value<double>("setup", "zstart"),
            astart = 1.0 / (1.0 + zstart),
            H0     = pcc->cosmo_param_["H0"],
            omegam = pcc->cosmo_param_["Omega_m"],
            omegaL = pcc->cosmo_param_["Omega_DE"];

        omegab_ = pcc->cosmo_param_["Omega_b"];
        
        uint32_t ngrid = cf_.get_value<int>("setup", "GridRes");

        bUseSPT_ = cf_.get_value_safe<bool>("output", "grafic_use_SPT", false);
        levelmin_ = uint32_t(std::log2(double(ngrid)) + 1e-6);

        if ( 1ul<<levelmin_ != ngrid )
        {
            music::elog << interface_name_ << " RAMSES requires setup/GridRes to be power of 2!" << std::endl;
            abort();
        }

        bhavebaryons_ = cf_.get_value_safe<bool>("setup", "baryons", false);

        header_.n1 = ngrid;
        header_.n2 = ngrid;
        header_.n3 = ngrid;
        header_.dxini0 = boxlength / (H0 * 0.01) / ngrid;
        header_.xoff10 = 0;
        header_.xoff20 = 0;
        header_.xoff30 = 0;
        header_.astart0 = astart;
        header_.omega_m0 = omegam;
        header_.omega_l0 = omegaL;
        header_.h00 = H0;

        data_buf_.assign(ngrid * ngrid, 0.0f);

        lunit_ = boxlength;
        vunit_ = boxlength;
        munit_ = 1.0 / omegam; // ramses wants mass in units of critical

        // create directory structure
        dirname_ = this->fname_;
        remove(dirname_.c_str());
        mkdir(dirname_.c_str(), 0777);

        // write RAMSES namelist file? if so only with one task
        if (cf_.get_value_safe<bool>("output", "ramses_nml", true) && CONFIG::MPI_task_rank==0 )
        {
            write_ramses_namelist();
        }
    }

    output_type write_species_as(const cosmo_species &s) const
    {
        if (s == cosmo_species::baryon && !bUseSPT_)
            return output_type::field_eulerian;
        return output_type::field_lagrangian;
    }

    bool has_64bit_reals() const{ return false; }

	bool has_64bit_ids() const{ return false; }

    real_t position_unit() const { return lunit_; }

    real_t velocity_unit() const { return vunit_; }

    real_t mass_unit() const { return munit_; }

    void write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c);
};

std::string grafic2_output_plugin::get_file_name(const cosmo_species &s, const fluid_component &c) const
{
    std::string file_name, species_str;

    file_name = dirname_ + "/ic_";

    switch (s)
    {
    case cosmo_species::dm:
        species_str = "c";
        break;
    case cosmo_species::baryon:
        species_str = "b";
        break;
    case cosmo_species::neutrino:
        species_str = "n";
        break;
    default:
        break;
    }

    switch (c)
    {
    case fluid_component::density:
        file_name += "delta" + species_str;
        break;
    case fluid_component::vx:
        file_name += "vel" + species_str + "x";
        break;
    case fluid_component::vy:
        file_name += "vel" + species_str + "y";
        break;
    case fluid_component::vz:
        file_name += "vel" + species_str + "z";
        break;
    case fluid_component::dx:
        file_name += "pos" + species_str + "x";
        break;
    case fluid_component::dy:
        file_name += "pos" + species_str + "y";
        break;
    case fluid_component::dz:
        file_name += "pos" + species_str + "z";
        break;
    case fluid_component::mass:
        file_name += "mass" + species_str;
        break;
    default:
        break;
    }

    return file_name;
}

void grafic2_output_plugin::write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c)
{
    // ignore certain components
    if (s == cosmo_species::dm && c == fluid_component::density)
        return;
    if (s == cosmo_species::baryon && (c == fluid_component::dx || c == fluid_component::dy || c == fluid_component::dz || c == fluid_component::mass ))
        return;

    if (c == fluid_component::mass){
        music::wlog << "You selected perturbed particle masses. " << std::endl;
        music::wlog << "Make sure your version of RAMSES supports this!" << std::endl;
    }

    // get file name based on species and fluid component type
    std::string file_name = this->get_file_name(s, c);

    // serialize parallel write
    if (CONFIG::MPI_task_rank == 0)
    {
        unlink(file_name.c_str());
    }

    std::ofstream *pofs = nullptr;

    // write header or seek to end of file
    if (CONFIG::MPI_task_rank == 0)
    {
        pofs = new std::ofstream(file_name.c_str(), std::ios::binary|std::ios::app);
        uint32_t blocksz = sizeof(header);
        pofs->write(reinterpret_cast<const char *>(&blocksz), sizeof(int));
        pofs->write(reinterpret_cast<const char *>(&header_), blocksz);
        pofs->write(reinterpret_cast<const char *>(&blocksz), sizeof(int));
    }

    // check field size against buffer size...
    uint32_t ngrid = cf_.get_value<int>("setup", "GridRes");
    assert( g.global_size(0) == ngrid && g.global_size(1) == ngrid && g.global_size(2) == ngrid);
    assert( g.size(1) == ngrid && g.size(2) == ngrid);
    // write actual field slice by slice
    // std::cerr << write_rank << ">" << g.size(0) << " " << g.size(1) << " " << g.size(2) << std::endl;
    for (size_t i = 0; i < g.size(2); ++i)
    {   
        data_buf_.assign(ngrid * ngrid, 0.0f);

        for (unsigned j = 0; j < g.size(1); ++j)
        {
            for (unsigned k = 0; k < g.size(0); ++k)
            {
                data_buf_[j * ngrid + (k+g.local_0_start_)] = g.relem(k, j, i);
            }
        }
#if defined(USE_MPI)
        if( CONFIG::MPI_task_rank == 0 ) data_buf_write_.assign(ngrid*ngrid,0.0f);
        MPI_Reduce( &data_buf_[0], &data_buf_write_[0], ngrid*ngrid, MPI::get_datatype<float>(), MPI_SUM, 0, MPI_COMM_WORLD );
        if( CONFIG::MPI_task_rank == 0 ) data_buf_.swap(data_buf_write_);
#endif

        if( CONFIG::MPI_task_rank == 0 )
        {
            uint32_t blocksz = ngrid * ngrid * sizeof(float);
            pofs->write(reinterpret_cast<const char *>(&blocksz), sizeof(uint32_t));
            pofs->write(reinterpret_cast<const char *>(&data_buf_[0]), blocksz);
            pofs->write(reinterpret_cast<const char *>(&blocksz), sizeof(uint32_t));
        }
    }

    if( CONFIG::MPI_task_rank == 0 ){
        pofs->close();
        delete pofs;
    }

    music::ilog << interface_name_ << " : Wrote field to file \'" << file_name << "\'" << std::endl;
}

void grafic2_output_plugin::write_ramses_namelist(void) const
{
    //... also write the refinement options to a dummy namelist file
    char ff[256];
    sprintf(ff, "%s/ramses.nml", dirname_.c_str());

    std::ofstream ofst(ff, std::ios::trunc);

    // -- RUN_PARAMS -- //
    ofst
        << "&RUN_PARAMS\n"
        << "cosmo=.true.\n"
        << "pic=.true.\n"
        << "poisson=.true.\n";

    if (bhavebaryons_)
        ofst << "hydro=.true.\n";
    else
        ofst << "hydro=.false.\n";

    ofst
        << "nrestart=0\n"
        << "nremap=1\n"
        << "nsubcycle=1,2\n"
        << "ncontrol=1\n"
        << "verbose=.false.\n"
        << "/\n\n";

    // -- INIT_PARAMS -- //
    ofst
        << "&INIT_PARAMS\n"
        << "filetype=\'grafic\'\n"
        << "omega_b=" << omegab_ << "\n"
        << "initfile(1)=\'" << dirname_ << "\'\n"
        << "/\n\n";

    // initialize with settings for naddref additional levels of refinement
    unsigned naddref = 5;

    // -- AMR_PARAMS -- //
    ofst << "&AMR_PARAMS\n"
         << "levelmin=" << levelmin_ << "\n"
         << "levelmax=" << levelmin_ + naddref << "\n"
         << "nexpand=1";

    ofst << "\n"
         << "ngridmax=2000000\n"
         << "npartmax=3000000\n"
         << "/\n\n";

    ofst << "&REFINE_PARAMS\n"
         << "m_refine=" << 1 + naddref << "*8.,\n"
         << "/\n";

    music::ilog << interface_name_ << " wrote partial RAMSES namelist file \'" << fname_ << "\'" << std::endl;
}

namespace
{
  output_plugin_creator_concrete<grafic2_output_plugin> creator201("grafic2");
} // namespace
