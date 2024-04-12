/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2019-2023 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed.org for more information.

   This file is part of plumed, version 2.

   plumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   plumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with plumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#include "CLTool.h"
#include "core/CLToolRegister.h"
#include "tools/Communicator.h"
#include "tools/Tools.h"
#include "config/Config.h"
#include "tools/PlumedHandle.h"
#include "tools/Stopwatch.h"
#include "tools/Log.h"
#include "tools/DLLoader.h"
#include <cstdio>
#include <string>
#include <vector>
#include <fstream>
#include <atomic>
#include <csignal>
#include <cstdio>
#include <random>
#include <algorithm>
#include <chrono>

namespace PLMD {
namespace cltools {

//+PLUMEDOC TOOLS benchmark
/*
benchmark is a lightweight reimplementation of driver focused on running benchmarks

The main difference wrt driver is that it generates a trajectory in memory rather than reading it
from a file. This allows to better time the overhead of the plumed library, without including
the time needed to read the trajectory.

It is also possible to load a separate version of the plumed kernel. This enables running
benchmarks agaist previous plumed versions in a controlled setting, where systematic errors
in the comparison are minimized.

\par Examples

First, you should create a sample `plumed.dat` file for testing. For instance:

\plumedfile
WHOLEMOLECULES ENTITY0=1-10000
p: POSITION ATOM=1
RESTRAINT ARG=p.x KAPPA=1 AT=0
\endplumedfile

Then you can test the performance of this input with the following command:
\verbatim
plumed benchmark
\endverbatim

You can also test a different (older) version of PLUMED with the same input. To do so,
you should run
\verbatim
plumed-runtime benchmark --kernel /path/to/lib/libplumedKernel.so
\endverbatim

\warning It is necessary to use the `plumed-runtime` executable here to avoid conflicts between different
plumed versions. You will find it in your path if you are using the non installed version of plumed,
and in `$prefix/lib/plumed` if you installed plumed in $prefix,.

\par Comparing multiple versions

The best way to compare two versions of plumed on the same input is to pass multiple colon-separated kernels:

\verbatim
plumed-runtime benchmark --kernel /path/to/lib/libplumedKernel.so:/path2/to/lib/libplumedKernel.so:this
\endverbatim

Here `this` means the kernel of the version with which you are running the benchmark. This comparison runs the three
instances simultaneously (alternating them) so that systematic differences in the load of your machine will affect them
to the same extent.

In case the different versions require modified plumed.dat files, or if you simply want to compare
two different plumed input files that compute the same thing, you can also use multiple plumed input files:

\verbatim
plumed-runtime benchmark --kernel /path/to/lib/libplumedKernel.so:this --plumed plumed1.dat:plumed2.dat
\endverbatim

Similarly, you might want to run two different inputs using the same kernel, which can be obtained with:

\verbatim
plumed-runtime benchmark --plumed plumed1.dat:plumed2.dat
\endverbatim

\par Profiling

If you want to attach a profiler on the fly to the process, you might find it convenient to use `--nsteps -1`.
The simulation will run forever and can be interrupted with CTRL-C. When interrupted, the result of the timers
should be displayed anyway.
You can also run setting a maximum time with `--maxtime`.

If you run a profiler when testing multiple PLUMED versions you might be confused by which function is from
each version. It is recommended to recompile separate instances with a separate C++ namespace (`-DPLMD=PLUMED_version_1`)
so that you will be able to distinguish them. In addition, compiling with `CXXFLAGS="-g -O3"` will make the profiling
report more complete, likely including code lines.

\par MPI runs

You can run emulating a domain decomposition. This is done automatically if plumed has been compiled with MPI
and you run with `mpirun`

\verbatim
mpirun -np 4 plumed-runtime benchmark
\endverbatim

If you load separate PLUMED instances as discussed above, they should all be compiled against the same MPI version.
Notice that when using MPI signals (CTRL-C) might not work.

Since some of the data transfer could happen asynchronously, you might want to use the `--sleep` option
to simulate a lag between the `prepareCalc` and `performCalc` actions. This part of the calculation will not contribute
to timer, but will obviously slow down your test.

\par Output

In the output you will see the usual reports about timing produced by the internal
timers of the tested plumed instances.
In addition, this tool will monitor the timing externally, with some slightly different criterion:
- First, the initialization (construction of the input) will be shown with a separate timer,
  as well as the timing for the first step.
- Second, the timer corresponding to the calculation will be split in three parts, reporting
  execution of the first 20% (warm-up) and the next two blocks of 40% each.
- Finally, you might notice some discrepancy because some of the actions that are usually
  not expensive are not included in the internal timers. The external timer will
  thus provide a better estimate of the total elapsed time, including everything.

The internal timers are still useful to monitor what happens at the different stages
and, with \ref DEBUG `DETAILED_TIMERS`, what happens in each action.

When you run multiple version, a comparative analisys of the time spent within PLUMED in the various
instances will be done, showing the ratio between the total time and the time measured on the first
instance, which will act as a reference. Errors will be estimated with bootstrapping. The warm-up phase will be discarded for
this analysis.

*/
//+ENDPLUMEDOC

// We use an anonymous namespace here to avoid clashes with variables
// declared in other parts of the code
namespace {

//this is a sugar for changing idea fast about the rng
using generator = std::mt19937;
std::atomic<bool> signalReceived(false);

class SignalHandlerGuard {
public:
  SignalHandlerGuard(int signal, void (*newHandler)(int)) : signal_(signal) {
    // Store the current handler before setting the new one
    prevHandler_ = std::signal(signal, newHandler);
    if (prevHandler_ == SIG_ERR) {
      throw std::runtime_error("Failed to set signal handler");
    }
  }

  ~SignalHandlerGuard() {
    // Restore the previous handler upon destruction
    std::signal(signal_, prevHandler_);
  }

  // Delete copy constructor and assignment operator to prevent copying
  SignalHandlerGuard(const SignalHandlerGuard&) = delete;
  SignalHandlerGuard& operator=(const SignalHandlerGuard&) = delete;

private:
  int signal_;
  void (*prevHandler_)(int);
};

extern "C" void signalHandler(int signal) {
  if (signal == SIGINT) {
    signalReceived.store(true);
    fprintf(stderr, "Signal handler called\n");
  }
}

/// This base class contains members that are movable with default operations
struct KernelBase {
  std::string path;
  std::string plumed_dat;
  PlumedHandle handle;
  Stopwatch stopwatch;
  std::vector<long long int> timings;
  double comparative_timing=-1.0;
  double comparative_timing_error=-1.0;
  KernelBase(const std::string & path_,const std::string & plumed_dat_, Log* log_):
    path(path_),
    plumed_dat(plumed_dat_),
    handle([&]() {
    if(path_=="this") return PlumedHandle();
    else return PlumedHandle::dlopen(path.c_str());
  }()),
  stopwatch(*log_)
  {
  }
};

/// Local structure handling a kernel and the related timers.
/// This structure specifically contain the Log, which needs special treatment
/// in move semantics
struct Kernel :
  public KernelBase {
  Log* log=nullptr;
  Kernel(const std::string & path_,const std::string & the_plumed_dat, Log* log_):
    KernelBase(path_,the_plumed_dat,log_),
    log(log_)
  {
  }

  ~Kernel() {
    if(log) {
      (*log)<<"\n";
      (*log)        <<"Kernel:      "<<path<<"\n";
      (*log)        <<"Input:       "<<plumed_dat<<"\n";
      if(comparative_timing>0.0) {
        (*log).printf("Comparative: %.3f +- %.3f\n",comparative_timing,comparative_timing_error);
      }
    }
  }

  Kernel(Kernel && other) noexcept:
    KernelBase(std::move(other)),
    log(other.log)
  {
    other.log=nullptr; // ensure no log is done in the moved away object
  }

  Kernel & operator=(Kernel && other) noexcept
  {
    if(this != &other) {
      KernelBase::operator=(std::move(other));
      log=other.log;
      other.log=nullptr; // ensure no log is done in the moved away object
    }
    return *this;
  }
};

namespace benchDistributions {
///Acts as a template for any distribution
struct AtomDistribution {
  virtual void operator()(std::vector<Vector>& posToUpdate, unsigned, generator&)=0;
};

struct theLine:AtomDistribution {
  void operator()(std::vector<Vector>& posToUpdate, unsigned step, generator&) override {
    std::generate(posToUpdate.begin(),posToUpdate.end(),[j=0u,step]() {
      return Vector(step*j, step*j+1, step*j+2);
    });
  }
};

struct uniformSphere:AtomDistribution {
  void operator()(std::vector<Vector>& posToUpdate, unsigned step, generator& rng) override {
    constexpr double rmin=0.0;
    constexpr double rminCub =rmin*rmin*rmin;
    //giving more or less a cubic udm of volume for each atom: V=nat
    const double rmaxCub= (3.0/(4.0*PLMD::pi)) * posToUpdate.size();
    using realDistr = std::uniform_real_distribution<double>;
    realDistr rndRho (0.0,  (rmaxCub-rminCub));
    realDistr rndTheta (-1, 1.); //(goes in 1-2*rnd(0,1)
    realDistr rndPhi (0, 2.0 * PLMD::pi) ;

    std::generate(posToUpdate.begin(),posToUpdate.end(),[&]() {
      double rho  = std::cbrt (rminCub + rndRho (rng));
      double theta=std::acos (rndTheta (rng));
      double phi  = rndPhi (rng);
      return Vector (
               rho * sin (theta) * cos (phi),
               rho * sin (theta) * sin (phi),
               rho * cos (theta));
    });
  }
};

struct uniformCube:AtomDistribution {
  void operator()(std::vector<Vector>& posToUpdate, unsigned step, generator& rng) override {
    //giving more or less a cubic udm of volume for each atom: V = nat
    const double rmax = std::cbrt(static_cast<double>(posToUpdate.size()));
    using realDistr = std::uniform_real_distribution<double>;
    realDistr rndR (0.0,  rmax);

    std::generate(posToUpdate.begin(),posToUpdate.end(),[&]() {
      return Vector (rndR(rng),rndR(rng),rndR(rng));
    });
  }
};
} //namespace  benchDistributions
class Benchmark:
  public CLTool
{
  std::unique_ptr<benchDistributions::AtomDistribution> distribution;
public:
  static void registerKeywords( Keywords& keys );
  explicit Benchmark(const CLToolOptions& co );
  int main(FILE* in, FILE*out,Communicator& pc) override;
  std::string description()const override {
    return "run a calculation with a fixed trajectory to find bottlenecks in PLUMED";
  }
};

PLUMED_REGISTER_CLTOOL(Benchmark,"benchmark")

void Benchmark::registerKeywords( Keywords& keys ) {
  CLTool::registerKeywords( keys );
  keys.add("compulsory","--plumed","plumed.dat","convert the input in this file to the html manual");
  keys.add("compulsory","--kernel","this","colon separated path(s) to kernel(s)");
  keys.add("compulsory","--natoms","100000","the number of atoms to use for the simulation");
  keys.add("compulsory","--nsteps","2000","number of steps of MD to perform (-1 means forever)");
  keys.add("compulsory","--maxtime","-1","maximum number of seconds (-1 means forever)");
  keys.add("compulsory","--sleep","0","number of seconds of sleep, mimicking MD calculation");
  keys.add("compulsory","--atom-distribution","line","the kind of possible atomic displacement at each step");
  keys.addFlag("--domain-decomposition",false,"simulate domain decomposition, implies --shuffle");
  keys.addFlag("--shuffled",false,"reshuffle atoms");
}

Benchmark::Benchmark(const CLToolOptions& co ):
  CLTool(co)
{
  inputdata=commandline;
}


int Benchmark::main(FILE* in, FILE*out,Communicator& pc) {

  generator rng; // deterministic initialization to avoid issues with MPI

  struct FileDeleter {
    void operator()(FILE*f) const noexcept {
      if(f) std::fclose(f);
    }
  };

  std::unique_ptr<FILE,FileDeleter> log_dev_null{std::fopen("/dev/null","w")};

  Log log;
  if(pc.Get_rank()==0) {
    log.link(out);
  } else {
    log.link(log_dev_null.get());
  }
  log.setLinePrefix("BENCH:  ");

  std::vector<Kernel> kernels;

  // perform comparative analysis
  // ensure that kernels vector is destroyed from last to first element upon exit
  auto kernels_deleter=[&log](auto f) {
    if(!f) {
      return;
    }
    if(f->empty()) {
      return;
    }
    generator deleterRng;

    const auto size=f->back().timings.size();
    //B are the bootstrap iterations
    constexpr int B=200;
    const size_t numblocks=size;
    // for some reasons, blocked bootstrap underestimates error
    // For now I keep it off. If I remove it, the code below could be simplified
    // if(numblocks>20) numblocks=20;
    const auto blocksize=size/numblocks;

    if(f->size()<2) {
      log<<"Single run, skipping comparative analysis\n";
    } else if(size<10) {
      log<<"Too small sample, skipping comparative analysis\n";
    } else try {

        log<<"Running comparative analysis, "<<numblocks<<" blocks with size "<<blocksize<<"\n";

        std::vector<std::size_t> choice(size);
        std::uniform_int_distribution<> distrib(0, numblocks-1);
        std::vector<std::vector<long long int>> blocks(f->size());

        { int i=0;
          for(auto it = f->rbegin(); it != f->rend(); ++it,++i) {
            size_t l=0;
            blocks[i].assign(numblocks,0);
            for(auto j=0ULL; j<numblocks; j++) {
              for(auto k=0ULL; k<blocksize; k++) {
                plumed_assert(l<it->timings.size());
                blocks[i][j]+=it->timings[l];
                l++;
              }
            }
          }
        }

        std::vector<std::vector<double>> ratios(f->size());
        for(auto & r : ratios) {
          //B are the bootstrap iterations
          r.resize(B);
        }

        //B are the bootstrap iterations
        for(unsigned b=0; b<B; b++) {
          for(auto & c : choice) c=distrib(deleterRng);
          long long int reference=0;
          for(auto & c : choice) {
            reference+=blocks[0][c];
          }
          for(auto i=0ULL; i<blocks.size(); i++) {
            long long int estimate=0;
            // this would lead to separate bootstrap samples for each estimate:
            // for(auto & c : choice){c=distrib(deleterRng);}
            for(auto & c : choice) {
              estimate+=blocks[i][c];
            }
            ratios[i][b]=double(estimate)/double(reference);
          }
        }

        {
          int i=0;
          for(auto it = f->rbegin(); it != f->rend(); ++it,++i) {
            double sum=0.0;
            double sum2=0.0;
            for(auto r : ratios[i]) {
              sum+=r;
              sum2+=r*r;
            }
            //B are the bootstrap iterations
            it->comparative_timing=sum/B;
            it->comparative_timing_error=std::sqrt(sum2/B-sum*sum/(B*B));
          }
        }

      } catch(std::exception & e) {
        log<<"Unexpected error during comparative analysis\n";
        log<<e.what()<<"\n";
      }
    while(!f->empty()) f->pop_back();

  };
  std::unique_ptr<decltype(kernels),decltype(kernels_deleter)> kernels_deleter_obj(&kernels,kernels_deleter);


  // construct the kernels vector:
  {
    std::vector<std::string> allpaths;

    {
      std::string paths;
      parse("--kernel",paths);
      allpaths=Tools::getWords(paths,":");
    }

    std::vector<std::string> allplumed;
    {
      std::string paths;
      parse("--plumed",paths);
      allplumed=Tools::getWords(paths,":");
    }

    plumed_assert(allplumed.size()>0 && allpaths.size()>0);

    // this check only works on MacOS
#if defined(__APPLE__)
    // if any of the paths if different from "this", we check if libplumed was loaded locally to avoid conflicts.
    if(std::any_of(allpaths.begin(),allpaths.end(),[](auto value) {return value != "this";})) {
      if(DLLoader::isPlumedGlobal()) {
        plumed_error()<<"It looks like libplumed is loaded in the global namespace, you cannot load a different version of the kernel\n"
                      <<"Please make sure you use the plumed-runtime executable and that the env var PLUMED_LOAD_NAMESPACE is not set to GLOBAL";
      }
    }
#endif

    if(allplumed.size()>1 && allpaths.size()>1 && allplumed.size() != allpaths.size()) {
      plumed_error() << "--kernel and --plumed should have either one element or the same number of elements";
    }

    if(allplumed.size()>1 && allpaths.size()==1) for(unsigned i=1; i<allplumed.size(); i++) allpaths.push_back(allpaths[0]);
    if(allplumed.size()==1 && allpaths.size()>1) for(unsigned i=1; i<allpaths.size(); i++) allplumed.push_back(allplumed[0]);

    for(unsigned i=0; i<allpaths.size(); i++) kernels.emplace_back(allpaths[i],allplumed[i],&log);
  }

  // reverse order so that log happens in the forward order:
  std::reverse(kernels.begin(),kernels.end());

  // read other flags:
  bool shuffled=false;
  parseFlag("--shuffled",shuffled);
  int nf; parse("--nsteps",nf);
  unsigned natoms; parse("--natoms",natoms);

  double maxtime; parse("--maxtime",maxtime);

  bool domain_decomposition=false;
  parseFlag("--domain-decomposition",domain_decomposition);
  if(pc.Get_size()>1) domain_decomposition=true;
  if(domain_decomposition) shuffled=true;

  double timeToSleep;
  parse("--sleep",timeToSleep);

  std::vector<int> shuffled_indexes;

  { std::string atomicDistr;
    parse("--atom-distribution",atomicDistr);
    if(atomicDistr == "line") {
      distribution = std::make_unique<benchDistributions::theLine>();
    } else if (atomicDistr == "cube") {
      distribution = std::make_unique<benchDistributions::uniformCube>();
    } else if (atomicDistr == "line") {
      distribution = std::make_unique<benchDistributions::uniformSphere>();
    }
  }



  const auto initial_time=std::chrono::high_resolution_clock::now();

  for(auto & k : kernels) {
    auto & p(k.handle);
    auto sw=k.stopwatch.startStop("A Initialization");
    if(Communicator::plumedHasMPI() && domain_decomposition) p.cmd("setMPIComm",&pc.Get_comm());
    p.cmd("setRealPrecision",(int)sizeof(double));
    p.cmd("setMDLengthUnits",1.0);
    p.cmd("setMDChargeUnits",1.0);
    p.cmd("setMDMassUnits",1.0);
    p.cmd("setMDEngine","benchmarks");
    p.cmd("setTimestep",1.0);
    p.cmd("setPlumedDat",k.plumed_dat.c_str());
    p.cmd("setLog",out);
    p.cmd("setNatoms",natoms);
    p.cmd("init");
  }

  std::vector<double> cell( 9 ), virial( 9 );
  std::vector<Vector> pos( natoms ), forces( natoms );
  std::vector<double> masses( natoms, 1 ), charges( natoms, 0 );

  if(shuffled) {
    shuffled_indexes.resize(natoms);
    for(unsigned i=0; i<natoms; i++) shuffled_indexes[i]=i;
    std::shuffle(shuffled_indexes.begin(),shuffled_indexes.end(),rng);
  }

  // non owning pointers, used for shuffling the execution order
  std::vector<Kernel*> kernels_ptr;
  for(unsigned i=0; i<kernels.size(); i++) kernels_ptr.push_back(&kernels[i]);

  int plumedStopCondition=0;
  bool fast_finish=false;
  int part=0;

  log<<"Starting MD loop\n";
  log<<"Use CTRL+C to stop at any time and collect timers (not working in MPI runs)\n";
  // trap signals:
  SignalHandlerGuard sigIntGuard(SIGINT, signalHandler);


  for(int step=0; nf<0 || step<nf; ++step) {
    std::shuffle(kernels_ptr.begin(),kernels_ptr.end(),rng);
    (*distribution)(pos,step,rng);
    double* pos_ptr;
    double* for_ptr;
    double* charges_ptr;
    double* masses_ptr;
    int* indexes_ptr=nullptr;
    int n_local_atoms;

    if(domain_decomposition) {
      const auto nproc=pc.Get_size();
      const auto nn=natoms/nproc;
      //using int to remove warning, MPI don't work with unsigned
      int excess=natoms%nproc;
      const auto myrank=pc.Get_rank();
      auto shift=0;
      n_local_atoms=nn;
      if(myrank<excess) n_local_atoms+=1;
      for(int i=0; i<myrank; i++) {
        shift+=nn;
        if(i<excess) shift+=1;
      }
      pos_ptr=&pos[shift][0];
      for_ptr=&forces[shift][0];
      charges_ptr=&charges[shift];
      masses_ptr=&masses[shift];
      indexes_ptr=shuffled_indexes.data()+shift;
    } else {
      pos_ptr=&pos[0][0];
      for_ptr=&forces[0][0];
      charges_ptr=&charges[0];
      masses_ptr=&masses[0];
      n_local_atoms=natoms;
      indexes_ptr=shuffled_indexes.data();
    }

    const char* sw_name;
    if(part==0)      sw_name="B0 First step";
    else if(part==1) sw_name="B1 Warm-up";
    else if(part==2) sw_name="B2 Calculation part 1";
    else             sw_name="B3 Calculation part 2";


    for(unsigned i=0; i<kernels_ptr.size(); i++) {
      auto & p(kernels_ptr[i]->handle);

      {
        auto sw=kernels_ptr[i]->stopwatch.startPause(sw_name);
        p.cmd("setStep",step);
        p.cmd("setStopFlag",&plumedStopCondition);
        p.cmd("setForces",for_ptr,n_local_atoms*3);
        p.cmd("setBox",&cell[0],9);
        p.cmd("setVirial",&virial[0],9);
        p.cmd("setPositions",pos_ptr,n_local_atoms*3);
        p.cmd("setMasses",masses_ptr,n_local_atoms);
        p.cmd("setCharges",charges_ptr,n_local_atoms);
        if(shuffled) {
          p.cmd("setAtomsNlocal",n_local_atoms);
          p.cmd("setAtomsGatindex",indexes_ptr,n_local_atoms);
        }
        p.cmd("prepareCalc");
      }

      // mimick MD calculation here
      {
        unsigned k=0;
        auto start=std::chrono::high_resolution_clock::now();
        while(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-start).count()<(long long int)1e9*timeToSleep) k+=i*i;
        std::fprintf(log_dev_null.get(),"%u",k);
      }

      {
        auto sw=kernels_ptr[i]->stopwatch.startStop(sw_name);
        p.cmd("performCalc");
      }

      if(kernels_ptr.size()>1 && part>1) kernels_ptr[i]->timings.push_back(kernels_ptr[i]->stopwatch.getLastCycle(sw_name));
      if(plumedStopCondition || signalReceived.load()) fast_finish=true;
    }
    auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-initial_time).count();
    if(part==0) part=1;
    if(part<2) {
      if((maxtime>0 && elapsed>(long long int)(0.2*1e9*maxtime)) || (nf>0 && step+1>=nf/5) || (maxtime<0 && nf<0 && step+1>=100)) {
        part=2;
        log<<"Warm-up completed\n";
      }
    }
    if(part<3) {
      if((maxtime>0 && elapsed>(long long int)(0.6*1e9*maxtime)) || (nf>0 && step+1>=3*nf/5)) {
        part=3;
        log<<"60% completed\n";
      }
    }

    if(maxtime>0 && elapsed>(long long int)(1e9*maxtime)) fast_finish=true;

    {
      unsigned tmp=fast_finish;
      pc.Bcast(tmp,0);
      fast_finish=tmp;
    }
    if(fast_finish) break;
  }

  return 0;
}

} // namespace unnamed
} // namespace cltools
} // namespace PLMD
