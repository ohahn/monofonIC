#include <cmath>
#include <complex>
#include <iostream>
#include <fstream>
#include <thread>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include <general.hh>
#include <ic_generator.hh>


// initialise with "default" values
namespace CONFIG{
    int  MPI_thread_support = -1;
    int  MPI_task_rank = 0;
    int  MPI_task_size = 1;
    bool MPI_ok = false;
    bool MPI_threads_ok = false;
    bool FFTW_threads_ok = false;
    int  num_threads = 1;
}


#include "system_stat.hh"

int main( int argc, char** argv )
{
    csoca::Logger::SetLevel(csoca::LogLevel::Info);
    // csoca::Logger::SetLevel(csoca::LogLevel::Debug);

    //------------------------------------------------------------------------------
    // initialise MPI 
    //------------------------------------------------------------------------------
    
#if defined(USE_MPI)
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &CONFIG::MPI_thread_support);
    CONFIG::MPI_threads_ok = CONFIG::MPI_thread_support >= MPI_THREAD_FUNNELED;
    MPI_Comm_rank(MPI_COMM_WORLD, &CONFIG::MPI_task_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &CONFIG::MPI_task_size);
    CONFIG::MPI_ok = true;

    // set up lower logging levels for other tasks
    if( CONFIG::MPI_task_rank!=0 )
    {
        csoca::Logger::SetLevel(csoca::LogLevel::Error);
    }
#endif

    csoca::ilog << "-----------------------------------------------------------------------------" << std::endl
                << ">> FastLPT v0.1 >>" << std::endl
                << "-----------------------------------------------------------------------------" << std::endl;
    

    //------------------------------------------------------------------------------
    // Parse command line options
    //------------------------------------------------------------------------------

    if (argc != 2)
    {
        // print_region_generator_plugins();
        print_TransferFunction_plugins();
        print_RNG_plugins();
        print_output_plugins();

        csoca::elog << "In order to run, you need to specify a parameter file!" << std::endl;
        exit(0);
    }

    // open the configuration file 
    ConfigFile the_config(argv[1]);

    //------------------------------------------------------------------------------
    // Set up FFTW
    //------------------------------------------------------------------------------

#if defined(USE_FFTW_THREADS)
  #if defined(USE_MPI)
    if (CONFIG::MPI_threads_ok)
        CONFIG::FFTW_threads_ok = FFTW_API(init_threads)();
  #else
    CONFIG::FFTW_threads_ok = FFTW_API(init_threads)();
  #endif 
#endif

#if defined(USE_MPI)
    FFTW_API(mpi_init)();
#endif

    CONFIG::num_threads = the_config.GetValueSafe<unsigned>("execution", "NumThreads",std::thread::hardware_concurrency());
    
#if defined(USE_FFTW_THREADS)
    if (CONFIG::FFTW_threads_ok)
        FFTW_API(plan_with_nthreads)(CONFIG::num_threads);
#endif

    //------------------------------------------------------------------------------
    // Set up OpenMP
    //------------------------------------------------------------------------------

#if defined(_OPENMP)
    omp_set_num_threads(CONFIG::num_threads);
#endif

    //------------------------------------------------------------------------------
    // Write code configuration to screen
    //------------------------------------------------------------------------------
#if defined(USE_MPI)
    csoca::ilog << std::setw(40) << std::left << "MPI is enabled" << " : " << "yes (" << CONFIG::MPI_task_size << " tasks)" << std::endl;
#else
    csoca::ilog << std::setw(40) << std::left << "MPI is enabled" << " : " << "no" << std::endl;
#endif
    csoca::ilog << std::setw(40) << std::left << "MPI supports multi-threading" << " : " << (CONFIG::MPI_threads_ok? "yes" : "no") << std::endl;
    csoca::ilog << std::setw(40) << std::left << "Available HW threads / task" << " : " << std::thread::hardware_concurrency() << " (" << CONFIG::num_threads << " used)" << std::endl;
    csoca::ilog << std::setw(40) << std::left << "FFTW supports multi-threading" << " : " << (CONFIG::FFTW_threads_ok? "yes" : "no") << std::endl;
    csoca::ilog << std::setw(40) << std::left << "FFTW mode" << " : ";
#if defined(FFTW_MODE_PATIENT)
	csoca::ilog << "FFTW_PATIENT" << std::endl;
#elif defined(FFTW_MODE_MEASURE)
    csoca::ilog << "FFTW_MEASURE" << std::endl;
#else
	csoca::ilog << "FFTW_ESTIMATE" << std::endl;
#endif

    SystemStat::Memory mem;

    unsigned availpmem = mem.get_AvailMem()/1024/1024;
    unsigned usedpmem = mem.get_UsedMem()/1024/1024;
    unsigned maxpmem = availpmem, minpmem = availpmem;
    unsigned maxupmem = usedpmem, minupmem = usedpmem;
    
#if defined(USE_MPI)
    unsigned temp = 0;
    MPI_Allreduce(&minpmem,&temp,1,MPI_UNSIGNED,MPI_MIN,MPI_COMM_WORLD);  minpmem = temp;
    MPI_Allreduce(&maxpmem,&temp,1,MPI_UNSIGNED,MPI_MAX,MPI_COMM_WORLD);  maxpmem = temp;
    MPI_Allreduce(&minupmem,&temp,1,MPI_UNSIGNED,MPI_MIN,MPI_COMM_WORLD); minupmem = temp;
    MPI_Allreduce(&maxupmem,&temp,1,MPI_UNSIGNED,MPI_MAX,MPI_COMM_WORLD); maxupmem = temp;
#endif
    csoca::ilog << std::setw(40) << std::left << "Total system memory (phys)" << " : " << mem.get_TotalMem()/1024/1024 << " Mb" << std::endl;
    csoca::ilog << std::setw(40) << std::left << "Used system memory (phys)" << " : " << "Max: " << maxupmem << " Mb, Min: " << minupmem << " Mb" << std::endl;
    csoca::ilog << std::setw(40) << std::left << "Available system memory (phys)" << " : " <<  "Max: " << maxpmem << " Mb, Min: " << minpmem << " Mb" << std::endl;
    
    //--------------------------------------------------------------------
    // Initialise plug-ins
    //--------------------------------------------------------------------
    try
    {
        ic_generator::Initialise( the_config );
    }catch(...){
        csoca::elog << "Problem during initialisation. See error(s) above. Exiting..." << std::endl;
        #if defined(USE_MPI) 
        MPI_Finalize();
        #endif
        return 1;
    }

    ///////////////////////////////////////////////////////////////////////
    // do the job...
    ///////////////////////////////////////////////////////////////////////
    ic_generator::Run( the_config );
    ///////////////////////////////////////////////////////////////////////

#if defined(USE_MPI)
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif

    csoca::ilog << "Done." << std::endl;

    return 0;
}
