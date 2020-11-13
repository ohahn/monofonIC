// This file is part of monofonIC (MUSIC2)
// A software package to generate ICs for cosmological simulations
// Copyright (C) 2020 by Oliver Hahn
// Copyright (C) 2012 by Jan Frederik Engels
// Copyright (C) 2020 by Bodo Schwabe
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

#ifdef ENABLE_AMREX

#include <output_plugin.hh>

#include <AMReX_VisMF.H>
#include <AMReX_Box.H>
#include <AMReX_RealBox.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Utility.H>
#include <AMReX_Array.H>
#include <AMReX_BoxArray.H>
#include <AMReX_FabArray.H>
#include <AMReX_MultiFab.H>

#define BL_SPACEDIM 3

class nyx_output_plugin : public output_plugin
{
protected:

  struct sim_header{
    std::vector<int> dimensions;
    std::vector<int> offset;
    float a_start;
    float dx;
    float h0;
    float omega_b;
    float omega_m;
    float omega_v;
    float vfact;
    float boxlength;
    int   particle_idx;
  };

  int n_data_items;
  std::vector<std::string> field_name;
  int f_lev;
  int ngrid;
  uint32_t levelmin_;
  uint32_t levelmax_;
  real_t lunit_, vunit_, munit_;
  std::vector<amrex::MultiFab*> mfs;
  std::vector<amrex::BoxArray> boxarrays;
  sim_header the_sim_header;

  int get_comp_idx(const cosmo_species &s, const fluid_component &c) const;


public:
  //constructor
  explicit nyx_output_plugin( config_file& cf, std::unique_ptr<cosmology::calculator> &pcc )
    : output_plugin(cf, pcc, "NYX/AMREX")
  {
    int argc=0;
    char **argv;
    amrex::Initialize(argc,argv);

    bool bhave_hydro = cf_.get_value<bool>("setup", "DoBaryons");
    
    if (bhave_hydro)
      n_data_items = 10;
    else
      n_data_items = 6;

    field_name.resize(n_data_items);
    if (bhave_hydro)
      {
	field_name[0] = "baryon_density";
	field_name[1] = "baryon_vel_x";
	field_name[2] = "baryon_vel_y";
	field_name[3] = "baryon_vel_z";
	field_name[4] = "dm_pos_x";
	field_name[5] = "dm_pos_y";
	field_name[6] = "dm_pos_z";
	field_name[7] = "dm_vel_x";
	field_name[8] = "dm_vel_y";
	field_name[9] = "dm_vel_z";
	the_sim_header.particle_idx = 4;
      }
    else
      {
	field_name[0] = "dm_pos_x";
	field_name[1] = "dm_pos_y";
	field_name[2] = "dm_pos_z";
	field_name[3] = "dm_vel_x";
	field_name[4] = "dm_vel_y";
	field_name[5] = "dm_vel_z";
	the_sim_header.particle_idx = 0;
      }

    ngrid = int(cf_.get_value<int>("setup", "GridRes"));
    f_lev = 0; //levelmax_-levelmin_;

    mfs.resize(1);
    amrex::BoxArray   domainBoxArray(1);
    amrex::IntVect    pdLo(0);
    amrex::IntVect    pdHi(ngrid-1);
    amrex::Box        probDomain(pdLo,pdHi);
    domainBoxArray.set(0, probDomain);
    domainBoxArray.maxSize(32);
    amrex::DistributionMapping dm {domainBoxArray};
    int ngrow(0);
    mfs[0] = new amrex::MultiFab(domainBoxArray, dm, n_data_items, ngrow);

    the_sim_header.dimensions.push_back( 1<<levelmin_ );
    the_sim_header.dimensions.push_back( 1<<levelmin_ );
    the_sim_header.dimensions.push_back( 1<<levelmin_ );

    the_sim_header.offset.push_back( 0 );
    the_sim_header.offset.push_back( 0 );
    the_sim_header.offset.push_back( 0 );

    the_sim_header.a_start		= 1.0/(1.0+cf.get_value<double>("setup","zstart"));
    the_sim_header.dx			= cf.get_value<double>("setup","BoxLength")/the_sim_header.dimensions[0]/(pcc->cosmo_param_["H0"]*0.01);
    the_sim_header.boxlength=cf.get_value<double>("setup","BoxLength");
    the_sim_header.h0			= pcc->cosmo_param_["H0"]*0.01;

    if( bhave_hydro )
      the_sim_header.omega_b		= pcc->cosmo_param_["Omega_b"];
    else
      the_sim_header.omega_b		= 0.0;

    the_sim_header.omega_m		= pcc->cosmo_param_["Omega_m"];
    the_sim_header.omega_v		= pcc->cosmo_param_["Omega_DE"];
    //!!!WARING: currently vfact=0!!!
    the_sim_header.vfact		= pcc->cosmo_param_["vfact"]*the_sim_header.h0;

    //Fix these!
    lunit_ = 1.0;
    vunit_ = 1.0;
    munit_ = 1.0;

  }

  //destructor
  virtual ~nyx_output_plugin()
  {
    std::string FullPath = fname_;
    if (!amrex::UtilCreateDirectory(FullPath, 0755))
      amrex::CreateDirectoryFailed(FullPath);
    if (!FullPath.empty() && FullPath[FullPath.size()-1] != '/')
      FullPath += '/';
    FullPath += "Header";
    std::ofstream Header(FullPath.c_str());
    
    for(int lev=0; lev <= f_lev; lev++)
      {
	writeLevelPlotFile (	fname_,
				Header,
				amrex::VisMF::OneFilePerCPU,
				lev);
      }
    Header.close();
    
    writeGridsFile(fname_);
    writeInputsFile();
  }
  
  
  void write_particle_data(const particle::container &pc, const cosmo_species &s, double Omega_species );

  void write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c);
  
  output_type write_species_as(const cosmo_species &s) const
  {
    if (s == cosmo_species::baryon)
      return output_type::field_eulerian;
    return output_type::field_lagrangian;
  }
  
  bool has_64bit_reals() const{ return false; }
  
  bool has_64bit_ids() const{ return false; }
  
  real_t position_unit() const { return lunit_; }
  
  real_t velocity_unit() const { return vunit_; }
  
  real_t mass_unit() const { return munit_; }

  void writeInputsFile( void );

  void writeGridsFile (const std::string& dir);

  void writeLevelPlotFile (const std::string& dir, std::ostream& os, amrex::VisMF::How how, int level);
  
};

void nyx_output_plugin::write_grid_data(const Grid_FFT<real_t> &g, const cosmo_species &s, const fluid_component &c){

  if(s==cosmo_species::neutrino)
    return;
  if(s==cosmo_species::dm && (c==fluid_component::density || c==fluid_component::mass) )
    return;

  int comp = this->get_comp_idx(s, c);

  assert( g.global_size(0) == ngrid && g.global_size(1) == ngrid && g.global_size(2) == ngrid);
  assert( g.size(1) == ngrid && g.size(2) == ngrid);

  //construct amrex type data container mf
#ifdef USE_MPI
  amrex::Vector<int> pmap(CONFIG::MPI_task_size);
  amrex::BoxArray   domainBoxArray(CONFIG::MPI_task_size);
    
  int *xlo = (int *)malloc(sizeof(int) * CONFIG::MPI_task_size);
  MPI_Allgather(&g.get_global_range().x1_[0], 1, MPI_INT, xlo, 1, MPI_INT, MPI_COMM_WORLD);
  int *xhi = (int *)malloc(sizeof(int) * CONFIG::MPI_task_size);
  MPI_Allgather(&g.get_global_range().x2_[0], 1, MPI_INT, xhi, 1, MPI_INT, MPI_COMM_WORLD);
  
  for(int i=0; i<CONFIG::MPI_task_size; ++i){
    pmap[i]=i;
    amrex::IntVect lo(xlo[i],0,0);
    amrex::IntVect hi(xhi[i]-1,ngrid-1,ngrid-1);
    amrex::Box box(lo,hi);
    domainBoxArray.set(i, box);
  }
#else
  amrex::Vector<int> pmap(1);
  amrex::BoxArray   domainBoxArray(1);
  pmap[0]=0;
  amrex::IntVect lo(0,0,0);
  amrex::IntVect hi(ngrid-1,ngrid-1,ngrid-1);
  amrex::Box box(lo,hi);
  domainBoxArray.set(0, probDomain);
#endif

  amrex::DistributionMapping domainDistMap(pmap);
  boxarrays.push_back(domainBoxArray);
  amrex::MultiFab mf(domainBoxArray, domainDistMap, 1, 0);

  //write data to mf
  for(amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
    amrex::FArrayBox &myFab = mf[mfi];
    const amrex::Box& box = mfi.validbox();
    const int  *fab_lo = box.loVect();
    const int  *fab_hi = box.hiVect();

    for (int k = fab_lo[2], mk = 0; k <= fab_hi[2]; k++,mk++)
      for (int j = fab_lo[1], mj = 0; j <= fab_hi[1]; j++,mj++)
	for (int i = fab_lo[0], mi = 0; i <= fab_hi[0]; i++,mi++) {
	  
	  amrex::IntVect iv(i,j,k);
	  int idx = myFab.box().index(iv);
	  myFab.dataPtr(0)[idx] = g.relem(mi, mj, mk);
	  
	}
  }

  //copy to global data container
  (*mfs[0]).ParallelCopy(mf, 0, comp, 1);

}

int nyx_output_plugin::get_comp_idx(const cosmo_species &s, const fluid_component &c) const
{
  int comp=-1;

  if(cf_.get_value<bool>("setup", "DoBaryons")){
    switch(s){
    case cosmo_species::baryon:
      switch(c){
      case fluid_component::density:
	comp = 0;
	break;
      case fluid_component::vx:
	comp = 1;
	break;
      case fluid_component::vy:
	comp = 2;
	break;
      case fluid_component::vz:
	comp = 3;
	break;
      default:
      	music::wlog << "baryon fluid_component ignored by nyx output. " << std::endl;
	break;
      }
      break;
    case cosmo_species::dm:
      switch(c){
      case fluid_component::dx:
	comp = 4;
	break;
      case fluid_component::dy:
	comp = 5;
	break;
      case fluid_component::dz:
	comp = 6;
	break;
      case fluid_component::vx:
	comp = 7;
	break;
      case fluid_component::vy:
	comp = 8;
	break;
      case fluid_component::vz:
	comp = 9;
	break;
      default:
	music::wlog << "dm fluid_component ignored by nyx output. " << std::endl;
	break;
      }
      break;
    default:
      music::wlog << "cosmo species ignored by nyx output. " << std::endl;
      break;
    }
  }else{
    switch(s){
    case cosmo_species::dm:
      switch(c){
      case fluid_component::dx:
	comp = 0;
	break;
      case fluid_component::dy:
	comp = 1;
	break;
      case fluid_component::dz:
	comp = 2;
	break;
      case fluid_component::vx:
	comp = 3;
	break;
      case fluid_component::vy:
	comp = 4;
	break;
      case fluid_component::vz:
	comp = 5;
	break;
      default:
	music::wlog << "dm fluid_component ignored by nyx output. " << std::endl;
	break;
      }
      break;
    default:
      music::wlog << "cosmo species ignored by nyx output. " << std::endl;
      break;
    }  
  }

  return comp;
}

void nyx_output_plugin::write_particle_data(const particle::container &pc, const cosmo_species &s, double Omega_species ) {}

void nyx_output_plugin::writeGridsFile (const std::string& dir)
  {
    
    std::string myFname = dir;
    if (!myFname.empty() && myFname[myFname.size()-1] != '/')
      myFname += '/';
    myFname += "grids_file";
    
    std::ofstream os(myFname.c_str());
    
    os << f_lev << '\n';
    
    for (int lev = 1; lev <= f_lev; lev++)
      {
	os << boxarrays[lev].size() << '\n';
	boxarrays[lev].coarsen(2);
	for (int i=0; i < boxarrays[lev].size(); i++)
	  os << boxarrays[lev][i] << "\n";
      }
    os.close();
  }

void nyx_output_plugin::writeLevelPlotFile (const	std::string&	dir,
			   std::ostream&	os,
			   amrex::VisMF::How	how,
			   int		level)
{
  
  if (level == 0)
    {
      //
      // The first thing we write out is the plotfile type.
      //
      os << "MUSIC_for_Nyx_v0.1" << '\n';
      
      os << n_data_items << '\n';
      
      for (int i = 0; i < n_data_items; i++)
	os << field_name[i] << '\n';
      
      os << 3 << '\n';
      os << 0 << '\n';
      
      os << f_lev << '\n';
      
      for (int i = 0; i < BL_SPACEDIM; i++)
	os << 0 << ' '; //ProbLo
      os << '\n';
      for (int i = 0; i < BL_SPACEDIM; i++)
	os << the_sim_header.boxlength/the_sim_header.h0 << ' '; //ProbHi
      os << '\n';
      
      for (int i = 0; i < f_lev; i++)
	os << 2 << ' '; //refinement factor
      os << '\n';
      
      amrex::IntVect    pdLo(0,0,0);
      amrex::IntVect    pdHi(ngrid-1,ngrid-1,ngrid-1);
      for (int i = 0; i <= f_lev; i++) //Geom(i).Domain()
	{
	  amrex::Box        probDomain(pdLo,pdHi);
	  os << probDomain << ' ';
	  pdHi *= 2;
	  pdHi += 1;
	}
      os << '\n';
      
      for (int i = 0; i <= f_lev; i++) //level steps
	os << 0 << ' ';
      os << '\n';
      
      double dx = the_sim_header.boxlength/ngrid/the_sim_header.h0;
      for (int i = 0; i <= f_lev; i++)
	{
	  for (int k = 0; k < BL_SPACEDIM; k++)
	    os << dx << ' ';
	  os << '\n';
	  dx = dx/2.;
	}
      os << 0 << '\n';
      os << "0\n"; // Write bndry data.
    }
  
  //
  // Build the directory to hold the MultiFab at this level.
  // The name is relative to the directory containing the Header file.
  //
  static const std::string BaseName = "/Cell";
  
  std::string Level = amrex::Concatenate("Level_", level, 1);
  //
  // Now for the full pathname of that directory.
  //
  std::string FullPath = dir;
  if (!FullPath.empty() && FullPath[FullPath.size()-1] != '/')
    FullPath += '/';
  FullPath += Level;
  //
  // Only the I/O processor makes the directory if it doesn't already exist.
  //
  if (!amrex::UtilCreateDirectory(FullPath, 0755))
    amrex::CreateDirectoryFailed(FullPath);
  
  os << level << ' ' << boxarrays[level].size() << ' ' << 0 << '\n';
  os << 0 << '\n';
  
  double cellsize[3];
  double dx = the_sim_header.boxlength/ngrid/the_sim_header.h0;
  for (int n = 0; n < BL_SPACEDIM; n++)
    {
      cellsize[n] = dx;
    }
  for (int i = 0; i < level; i++)
    {
      for (int n = 0; n < BL_SPACEDIM; n++)
	{
	  cellsize[n] /= 2.;
	}
    }
  for (int i = 0; i < boxarrays[level].size(); ++i)
    {
      double problo[] = {0,0,0};
      amrex::RealBox gridloc = amrex::RealBox(boxarrays[level][i], cellsize, problo);
      for (int n = 0; n < BL_SPACEDIM; n++)
	os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
    }
  //
  // The full relative pathname of the MultiFabs at this level.
  // The name is relative to the Header file containing this name.
  // It's the name that gets written into the Header.
  //
  std::string PathNameInHeader = Level;
  PathNameInHeader += BaseName;
  os << PathNameInHeader << '\n';
  
  //
  // Use the Full pathname when naming the MultiFab.
  //
  std::string TheFullPath = FullPath;
  TheFullPath += BaseName;
  amrex::VisMF::Write((*mfs[level]),TheFullPath,how,true);
}

void nyx_output_plugin::writeInputsFile( void )
{
  //
  //before finalizing we write out an inputs and a probin file for Nyx.
  //
  std::ofstream inputs("inputs");
  std::ofstream probin("probin");
  
  //at first the fortran stuff...
  probin << "&fortin" << std::endl;
  probin << "  comoving_OmM = " << the_sim_header.omega_m << "d0" << std::endl;
  probin << "  comoving_OmB = " << the_sim_header.omega_b << "d0" << std::endl;
  probin << "  comoving_OmL = " << the_sim_header.omega_v << "d0" << std::endl;
  probin << "  comoving_h   = " << the_sim_header.h0      << "d0" << std::endl;
  probin << "/" << std::endl;
  probin << std::endl;
  
  //afterwards the cpp stuff...(for which we will need a template, which is read in by the code...)
  inputs << "nyx.final_a = 1.0 " << std::endl;
  inputs << "max_step = 100000 " << std::endl;
  inputs << "comoving_OmM = " << the_sim_header.omega_m << std::endl;
  inputs << "comoving_OmB = " << the_sim_header.omega_b << std::endl;
  inputs << "comoving_OmL = " << the_sim_header.omega_v << std::endl;
  inputs << "comoving_h   = " << the_sim_header.h0 << std::endl;
  inputs << "nyx.small_dens = 1e-4" << std::endl;
  inputs << "nyx.small_temp = 10" << std::endl;
  inputs << "nyx.cfl            = 0.9     # cfl number for hyperbolic system" << std::endl;
  inputs << "nyx.init_shrink    = 1.0     # scale back initial timestep" << std::endl;
  inputs << "nyx.change_max     = 1.05    # scale back initial timestep" << std::endl;
  inputs << "nyx.dt_cutoff      = 5.e-20  # level 0 timestep below which we halt" << std::endl;
  inputs << "nyx.sum_interval   = 1      # timesteps between computing mass" << std::endl;
  inputs << "nyx.v              = 2       # verbosity in Castro.cpp" << std::endl;
  inputs << "gravity.v             = 2       # verbosity in Gravity.cpp" << std::endl;
  inputs << "amr.v                 = 2       # verbosity in Amr.cpp" << std::endl;
  inputs << "mg.v                  = 2       # verbosity in Amr.cpp" << std::endl;
  inputs << "particles.v           = 2       # verbosity in Particle class" << std::endl;
  inputs << "amr.ref_ratio       = 2 2 2 2 2 2 2 2 " << std::endl;
  inputs << "amr.regrid_int      = 2 2 2 2 2 2 2 2 " << std::endl;
  inputs << "amr.initial_grid_file = init/grids_file" << std::endl;
  inputs << "amr.useFixedCoarseGrids = 1" << std::endl;
  inputs << "amr.check_file      = chk " << std::endl;
  inputs << "amr.check_int       = 10 " << std::endl;
  inputs << "amr.plot_file       = plt " << std::endl;
  inputs << "amr.plot_int        = 10 " << std::endl;
  inputs << "amr.derive_plot_vars = particle_count particle_mass_density pressure" << std::endl;
  inputs << "amr.plot_vars = ALL" << std::endl;
  inputs << "nyx.add_ext_src = 0" << std::endl;
  inputs << "gravity.gravity_type = PoissonGrav    " << std::endl;
  inputs << "gravity.no_sync      = 1              " << std::endl;
  inputs << "gravity.no_composite = 1              " << std::endl;
  inputs << "mg.bottom_solver = 1                  " << std::endl;
  inputs << "geometry.is_periodic =  1     1     1 " << std::endl;
  inputs << "geometry.coord_sys   =  0             " << std::endl;
  inputs << "amr.max_grid_size    = 32             " << std::endl;
  inputs << "nyx.lo_bc       =  0   0   0          " << std::endl;
  inputs << "nyx.hi_bc       =  0   0   0          " << std::endl;
  inputs << "nyx.do_grav  = 1                      " << std::endl;
  inputs << "nyx.do_dm_particles = 1               " << std::endl;
  inputs << "nyx.particle_init_type = Cosmological " << std::endl;
  inputs << "nyx.print_fortran_warnings = 0" << std::endl;
  inputs << "cosmo.initDirName  = init             " << std::endl;
  inputs << "nyx.particle_move_type = Gravitational" << std::endl;
  inputs << "amr.probin_file = probin              " << std::endl;
  inputs << "cosmo.ic-source = MUSIC               " << std::endl;
  
  if(cf_.contains_key( "setup", "blocking_factor"))
    inputs << "amr.blocking_factor = " << cf_.get_value<double>("setup","blocking_factor") << std::endl;
  else
    inputs << "amr.blocking_factor = 8             " << std::endl;
  
  inputs << "nyx.do_hydro = "<< (the_sim_header.omega_b>0?1:0) << std::endl;
  inputs << "amr.max_level       = " << f_lev << std::endl;
  inputs << "nyx.initial_z = " << 1/the_sim_header.a_start-1 << std::endl;
  inputs << "amr.n_cell           = " << ngrid << " " << ngrid << " " << ngrid << std::endl;
  inputs << "nyx.n_particles      = " << ngrid << " " << ngrid << " " << ngrid << std::endl;
  inputs << "geometry.prob_lo     = 0 0 0" << std::endl;
  
  double bl = the_sim_header.boxlength/the_sim_header.h0;
  inputs << "geometry.prob_hi     = " << bl << " " << bl << " " << bl << std::endl;
  
  
  probin.close();
  inputs.close();
  
}


namespace{
  output_plugin_creator_concrete<nyx_output_plugin> creator("nyx");
}

#endif //ENABLE_AMREX

