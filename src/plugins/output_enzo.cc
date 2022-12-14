// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2022 by Oliver Hahn
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

// #ifdef HAVE_HDF5

#include <unistd.h> // for unlink
#include <sys/types.h>
#include <sys/stat.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <vector>

#include <general.hh>
#include <output_plugin.hh>

#include "HDF_IO.hh"

#define MAX_SLAB_SIZE 268435456 // = 256 MBytes

class enzo_output_plugin : public output_plugin
{
protected:
  struct patch_header
  {
    int component_rank;
    size_t component_size;
    std::vector<int> dimensions;
    int rank;
    std::vector<int> top_grid_dims;
    std::vector<int> top_grid_end;
    std::vector<int> top_grid_start;
  };

  struct sim_header
  {
    std::vector<int> dimensions;
    std::vector<int> offset;
    float a_start;
    float dx;
    float h0;
    float omega_b;
    float omega_m;
    float omega_v;
    float vfact;
  };

  sim_header the_sim_header;
  bool bUseSPT_;

  void write_sim_header(std::string fname, const sim_header &h)
  {
    HDFWriteGroupAttribute(fname, "/", "Dimensions", h.dimensions);
    HDFWriteGroupAttribute(fname, "/", "Offset", h.offset);
    HDFWriteGroupAttribute(fname, "/", "a_start", h.a_start);
    HDFWriteGroupAttribute(fname, "/", "dx", h.dx);
    HDFWriteGroupAttribute(fname, "/", "h0", h.h0);
    HDFWriteGroupAttribute(fname, "/", "omega_b", h.omega_b);
    HDFWriteGroupAttribute(fname, "/", "omega_m", h.omega_m);
    HDFWriteGroupAttribute(fname, "/", "omega_v", h.omega_v);
    HDFWriteGroupAttribute(fname, "/", "vfact", h.vfact);
  }

  void write_patch_header(std::string fname, std::string dsetname, const patch_header &h)
  {
    HDFWriteDatasetAttribute(fname, dsetname, "Component_Rank", h.component_rank);
    HDFWriteDatasetAttribute(fname, dsetname, "Component_Size", h.component_size);
    HDFWriteDatasetAttribute(fname, dsetname, "Dimensions", h.dimensions);
    HDFWriteDatasetAttribute(fname, dsetname, "Rank", h.rank);
    HDFWriteDatasetAttribute(fname, dsetname, "TopGridDims", h.top_grid_dims);
    HDFWriteDatasetAttribute(fname, dsetname, "TopGridEnd", h.top_grid_end);
    HDFWriteDatasetAttribute(fname, dsetname, "TopGridStart", h.top_grid_start);
  }

  void dump_grid_data(std::string fieldname, const Grid_FFT<real_t> &g, double factor = 1.0, double add = 0.0)
  {
    char enzoname[256], filename[512];
    const int ngrid = cf_.get_value<int>("setup", "GridRes");
    {
      std::vector<int> ng{{ngrid, ngrid, ngrid}}, ng_fortran{{ngrid, ngrid, ngrid}};

      //... need to copy data because we need to get rid of the ghost zones
      //... write in slabs if data is more than MAX_SLAB_SIZE (default 128 MB)

      //... full 3D block size
      size_t all_data_size = (size_t)ng[0] * (size_t)ng[1] * (size_t)ng[2];

      //... write in slabs of MAX_SLAB_SIZE unless all_data_size is anyway smaller
      size_t max_slab_size = std::min((size_t)MAX_SLAB_SIZE / sizeof(double), all_data_size);

      //... but one slab hast to be at least the size of one slice
      max_slab_size = std::max(((size_t)ng[0] * (size_t)ng[1]), max_slab_size);

      //... number of slices in one slab
      size_t slices_in_slab = (size_t)((double)max_slab_size / ((size_t)ng[0] * (size_t)ng[1]));

      size_t nsz[3] = {size_t(ng[2]), size_t(ng[1]), size_t(ng[0])};

      // if (levelmin_ != levelmax_)
      //   sprintf(enzoname, "%s.%d", fieldname.c_str(), ilevel - levelmin_);
      // else
      sprintf(enzoname, "%s", fieldname.c_str());

      sprintf(filename, "%s/%s", fname_.c_str(), enzoname);

      HDFCreateFile(filename);
      write_sim_header(filename, the_sim_header);

#ifdef SINGLE_PRECISION
      //... create full array in file
      HDFHyperslabWriter3Ds<float> *slab_writer = new HDFHyperslabWriter3Ds<float>(filename, enzoname, nsz);

      //... create buffer
      float *data_buf = new float[slices_in_slab * (size_t)ng[0] * (size_t)ng[1]];
#else
      //... create full array in file
      HDFHyperslabWriter3Ds<double> *slab_writer = new HDFHyperslabWriter3Ds<double>(filename, enzoname, nsz);

      //... create buffer
      double *data_buf = new double[slices_in_slab * (size_t)ng[0] * (size_t)ng[1]];
#endif

      //... write slice by slice
      size_t slices_written = 0;
      while (slices_written < (size_t)ng[2])
      {
        slices_in_slab = std::min((size_t)ng[2] - slices_written, slices_in_slab);

#pragma omp parallel for
        for (int k = 0; k < (int)slices_in_slab; ++k)
          for (int j = 0; j < ng[1]; ++j)
            for (int i = 0; i < ng[0]; ++i)
              data_buf[(size_t)(k * ng[1] + j) * (size_t)ng[0] + (size_t)i] =
                  (add + g.relem(i, j, k + slices_written)) * factor;

        size_t count[3], offset[3];

        count[0] = slices_in_slab;
        count[1] = ng[1];
        count[2] = ng[0];

        offset[0] = slices_written;
        ;
        offset[1] = 0;
        offset[2] = 0;

        slab_writer->write_slab(data_buf, count, offset);
        slices_written += slices_in_slab;
      }

      //... free buffer
      delete[] data_buf;

      //... finalize writing and close dataset
      delete slab_writer;

      //... header data for the patch
      patch_header ph;

      ph.component_rank = 1;
      ph.component_size = (size_t)ng[0] * (size_t)ng[1] * (size_t)ng[2];
      ph.dimensions = ng;
      ph.rank = 3;

      ph.top_grid_dims.assign(3, ngrid);

      //... offset_abs is in units of the current level cell size

      // double rfac = 1.0 / (1 << (ilevel - levelmin_));

      ph.top_grid_start.push_back(0);
      ph.top_grid_start.push_back(0);
      ph.top_grid_start.push_back(0);

      ph.top_grid_end.push_back(ph.top_grid_start[0] + ngrid);
      ph.top_grid_end.push_back(ph.top_grid_start[1] + ngrid);
      ph.top_grid_end.push_back(ph.top_grid_start[2] + ngrid);

      write_patch_header(filename, enzoname, ph);
    }
  }

public:
  enzo_output_plugin(config_file &cf, std::unique_ptr<cosmology::calculator> &pcc)
      : output_plugin(cf, pcc, "ENZO")
  {
    if (CONFIG::MPI_task_size >1 ){
      music::elog << "ENZO output plugin currently does not support MPI. Please run using a single task only!" << std::endl;
      throw std::runtime_error("Error in enzo_output_plugin!");
    }
    
    if (CONFIG::MPI_task_rank == 0)
    {
      if (mkdir(fname_.c_str(), 0777))
      {
        perror(fname_.c_str());
        throw std::runtime_error("Error in enzo_output_plugin!");
      }
    }

    const bool bhave_hydro = cf_.get_value_safe<bool>("setup", "baryons", false);
    const uint32_t ngrid = cf_.get_value<int>("setup", "GridRes");
    bUseSPT_ = cf_.get_value_safe<bool>("output", "enzo_use_SPT", false);

    the_sim_header.dimensions.push_back(ngrid);
    the_sim_header.dimensions.push_back(ngrid);
    the_sim_header.dimensions.push_back(ngrid);

    the_sim_header.offset.push_back(0);
    the_sim_header.offset.push_back(0);
    the_sim_header.offset.push_back(0);

    the_sim_header.a_start = 1.0 / (1.0 + cf_.get_value<double>("setup", "zstart"));
    the_sim_header.dx = cf_.get_value<double>("setup", "BoxLength") / the_sim_header.dimensions[0] / (pcc->cosmo_param_["H0"] * 0.01); // not sure?!?

    the_sim_header.h0 = pcc->cosmo_param_["H0"] * 0.01;

    if (bhave_hydro)
      the_sim_header.omega_b = pcc->cosmo_param_["Omega_b"];
    else
      the_sim_header.omega_b = 0.0;

    the_sim_header.omega_m = pcc->cosmo_param_["Omega_m"];
    the_sim_header.omega_v = pcc->cosmo_param_["Omega_DE"];
    the_sim_header.vfact = pcc->get_vfact(the_sim_header.a_start); // TODO: check if should be multiplied by h (see below)

    // cf.getValue<double>("cosmology", "vfact") * the_sim_header.h0; //.. need to multiply by h, ENZO wants this factor for non h-1 units

    if (CONFIG::MPI_task_rank == 0)
    {
      write_enzo_parameter_file();
    }


  }

  ~enzo_output_plugin()
  {
  }

  bool has_64bit_reals() const { return false; }

  bool has_64bit_ids() const { return false; }

  real_t position_unit() const { return 1.0; }//lunit_; }

  real_t velocity_unit() const { return 1.0; }//vunit_; }

  real_t mass_unit() const { return 1.0; }//munit_; }

  output_type write_species_as(const cosmo_species &s) const
  {
    if (s == cosmo_species::baryon && !bUseSPT_)
      return output_type::field_eulerian;
    return output_type::field_lagrangian;
  }

  // void write_dm_mass(const grid_hierarchy &gh)
  // { /* do nothing, not needed */
  // }

  void write_enzo_parameter_file(void)
  { /* write the parameter file data */

    const bool bhave_hydro = the_sim_header.omega_b;
    const uint32_t ngrid = cf_.get_value<int>("setup", "GridRes");
    const double zstart = cf_.get_value<double>("setup", "zstart");
    const double boxlength = cf_.get_value<double>("setup", "BoxLength");

    // double refine_region_fraction = cf_.getValueSafe<double>("output", "enzo_refine_region_fraction", 0.8);
    char filename[256];

    // write out the refinement masks
    // dump_mask(gh);

    // write out a parameter file

    sprintf(filename, "%s/parameter_file.txt", fname_.c_str());

    std::ofstream ofs(filename, std::ios::trunc);

    ofs
        << "# Relevant Section of Enzo Paramter File (NOT COMPLETE!) \n"
        << "ProblemType                              = 30      // cosmology simulation\n"
        << "TopGridRank                              = 3\n"
        << "TopGridDimensions                        = " << ngrid << " " << ngrid << " " << ngrid << "\n"
        << "SelfGravity                              = 1       // gravity on\n"
        << "TopGridGravityBoundary                   = 0       // Periodic BC for gravity\n"
        << "LeftFaceBoundaryCondition                = 3 3 3   // same for fluid\n"
        << "RightFaceBoundaryCondition               = 3 3 3\n"
        << "RefineBy                                 = 2\n"
        << "\n"
        << "#\n";

    if (bhave_hydro)
      ofs
          << "CosmologySimulationOmegaBaryonNow        = " << the_sim_header.omega_b << "\n"
          << "CosmologySimulationOmegaCDMNow           = " << the_sim_header.omega_m - the_sim_header.omega_b << "\n";
    else
      ofs
          << "CosmologySimulationOmegaBaryonNow        = " << 0.0 << "\n"
          << "CosmologySimulationOmegaCDMNow           = " << the_sim_header.omega_m << "\n";

    if (bhave_hydro)
      ofs
          << "CosmologySimulationDensityName           = GridDensity\n"
          << "CosmologySimulationVelocity1Name         = GridVelocities_x\n"
          << "CosmologySimulationVelocity2Name         = GridVelocities_y\n"
          << "CosmologySimulationVelocity3Name         = GridVelocities_z\n";

    ofs
        << "CosmologySimulationCalculatePositions    = 1\n"
        << "CosmologySimulationParticleVelocity1Name = ParticleVelocities_x\n"
        << "CosmologySimulationParticleVelocity2Name = ParticleVelocities_y\n"
        << "CosmologySimulationParticleVelocity3Name = ParticleVelocities_z\n"
        << "CosmologySimulationParticleDisplacement1Name = ParticleDisplacements_x\n"
        << "CosmologySimulationParticleDisplacement2Name = ParticleDisplacements_y\n"
        << "CosmologySimulationParticleDisplacement3Name = ParticleDisplacements_z\n"
        << "\n"
        << "#\n"
        << "#  define cosmology parameters\n"
        << "#\n"
        << "ComovingCoordinates                      = 1       // Expansion ON\n"
        << "CosmologyOmegaMatterNow                  = " << the_sim_header.omega_m << "\n"
        << "CosmologyOmegaLambdaNow                  = " << the_sim_header.omega_v << "\n"
        << "CosmologyHubbleConstantNow               = " << the_sim_header.h0 << "     // in 100 km/s/Mpc\n"
        << "CosmologyComovingBoxSize                 = " << boxlength << "    // in Mpc/h\n"
        << "CosmologyMaxExpansionRate                = 0.015   // maximum allowed delta(a)/a\n"
        << "CosmologyInitialRedshift                 = " << zstart << "      //\n"
        << "CosmologyFinalRedshift                   = 0       //\n"
        << "GravitationalConstant                    = 1       // this must be true for cosmology\n"
        << "#\n"
        << "#\n"
        << "ParallelRootGridIO                       = 1\n"
        << "ParallelParticleIO                       = 1\n"
        << "PartitionNestedGrids                     = 1\n"
        << "CosmologySimulationNumberOfInitialGrids  = " << 1 << "\n";

    int num_prec = 10;

    // if (levelmax_ > 15)
    //   num_prec = 17;

    //... only for additionally refined grids
    for (unsigned ilevel = 0; ilevel < 1; ++ilevel)
    {
      ofs

          << "CosmologySimulationGridDimension[" << 1 + ilevel << "]      = "
          << std::setw(16) << ngrid << " "
          << std::setw(16) << ngrid << " "
          << std::setw(16) << ngrid << "\n"

          << "CosmologySimulationGridLeftEdge[" << 1 + ilevel << "]       = "
          << std::setw(num_prec + 6) << std::setprecision(num_prec) << 0.0 << " "
          << std::setw(num_prec + 6) << std::setprecision(num_prec) << 0.0 << " "
          << std::setw(num_prec + 6) << std::setprecision(num_prec) << 0.0 << "\n"

          << "CosmologySimulationGridRightEdge[" << 1 + ilevel << "]      = "
          << std::setw(num_prec + 6) << std::setprecision(num_prec) << 1.0 << " "
          << std::setw(num_prec + 6) << std::setprecision(num_prec) << 1.0 << " "
          << std::setw(num_prec + 6) << std::setprecision(num_prec) << 1.0 << "\n"

          << "CosmologySimulationGridLevel[" << 1 + ilevel << "]          = " << 1 + ilevel << "\n";
    }
  }

  void write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c)
  {
    std::string field_name;
    double field_mul = 1.0;
    double field_add = 0.0;

    switch (s)
    {
      case cosmo_species::dm:
        switch (c)
        {
          case fluid_component::dx:
            field_name = "ParticleDisplacements_x";
            break;
          case fluid_component::dy:
            field_name = "ParticleDisplacements_y";
            break;
          case fluid_component::dz:
            field_name = "ParticleDisplacements_z";
            break;
          case fluid_component::vx:
            field_name = "ParticleVelocities_x";
            field_mul = 1.0 / (1.225e2 * sqrt(the_sim_header.omega_m / the_sim_header.a_start));
            break;
          case fluid_component::vy:
            field_name = "ParticleVelocities_y";
            field_mul = 1.0 / (1.225e2 * sqrt(the_sim_header.omega_m / the_sim_header.a_start));
            break;
          case fluid_component::vz:
            field_name = "ParticleVelocities_z";
            field_mul = 1.0 / (1.225e2 * sqrt(the_sim_header.omega_m / the_sim_header.a_start));
            break;
          //... ignores: ...
          case fluid_component::mass:
            field_name = "ParticleMasses";
            // TODO this should be implemented! not sure it's supported by enzo, let's write it anyway, with units TBD
            break;
          case fluid_component::density:
            return;
        }
        break;

      case cosmo_species::baryon:
        switch (c)
        {
          case fluid_component::density:
            field_name = "GridDensity";
            field_mul = the_sim_header.omega_b / the_sim_header.omega_m;
            field_add = 1.0;
            break;
          case fluid_component::vx:
            field_name = "GridVelocities_x";
            field_mul = 1.0 / (1.225e2 * sqrt(the_sim_header.omega_m / the_sim_header.a_start));
            break;
          case fluid_component::vy:
            field_name = "GridVelocities_y";
            field_mul = 1.0 / (1.225e2 * sqrt(the_sim_header.omega_m / the_sim_header.a_start));
            break;
          case fluid_component::vz:
            field_name = "GridVelocities_z";
            field_mul = 1.0 / (1.225e2 * sqrt(the_sim_header.omega_m / the_sim_header.a_start));
            break;
          //... ignores: ...
          case fluid_component::dx:
          case fluid_component::dy:
          case fluid_component::dz:
          case fluid_component::mass:
            return;  
        }
        break;

      case cosmo_species::neutrino:
        return;
    }

    dump_grid_data(field_name, g, field_mul, field_add);
  }

  // void write_dm_velocity(int coord, const grid_hierarchy &gh)
  // {
  //   char enzoname[256];
  //   sprintf(enzoname, "ParticleVelocities_%c", (char)('x' + coord));

  //   double vunit = 1.0 / (1.225e2 * sqrt(the_sim_header.omega_m / the_sim_header.a_start));

  //   dump_grid_data(enzoname, gh, vunit);
  // }

  // void write_dm_position(int coord, const grid_hierarchy &gh)
  // {
  //   char enzoname[256];
  //   sprintf(enzoname, "ParticleDisplacements_%c", (char)('x' + coord));

  //   dump_grid_data(enzoname, gh);
  // }

  // void write_dm_potential(const grid_hierarchy &gh)
  // {
  // }

  // void write_gas_potential(const grid_hierarchy &gh)
  // {
  // }

  // void write_gas_velocity(int coord, const grid_hierarchy &gh)
  // {
  //   double vunit = 1.0 / (1.225e2 * sqrt(the_sim_header.omega_m / the_sim_header.a_start));

  //   char enzoname[256];
  //   sprintf(enzoname, "GridVelocities_%c", (char)('x' + coord));
  //   dump_grid_data(enzoname, gh, vunit);
  // }

  // void write_gas_position(int coord, const grid_hierarchy &gh)
  // {
  //   /* do nothing, not needed */
  // }

  // void write_gas_density(const grid_hierarchy &gh)
  // {

  //   char enzoname[256];
  //   sprintf(enzoname, "GridDensity");
  //   dump_grid_data(enzoname, gh, the_sim_header.omega_b / the_sim_header.omega_m, 1.0);
  // }

  void finalize(void)
  {
  }
};

namespace
{
  output_plugin_creator_concrete<enzo_output_plugin> creator1("ENZO");
}

// #endif // HAVE_HDF5
