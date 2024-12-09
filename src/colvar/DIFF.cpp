#include "Colvar.h"

#include "core/PlumedMain.h"
#include "core/ActionRegister.h"
#include "core/ActionAtomistic.h"

#include "tools/OpenMP.h"
#include "DIFFbase.h"

#include <torch/torch.h>
#include <torch/script.h>
#include <torch/cuda.h>

#include <fstream>
#include <omp.h>
#include <string>
#include <deque>


namespace PLMD {
    
    namespace colvar {

        class DIFF: public Colvar {
            
            private:
            
                vector<int> atomsNames;

				torch::Tensor refPeaksIntensitiesTensor;
                vector<double> refPeaksAngles, refPeaksIntensities;
              
                Lattice refLattice;
                Constants constants;
				
				bool doGrad, doBias;
				int gap;
				
				vector<Vector> derivatives;
				
				ForceType forceType;
				double forceCoeff;
				
				
				
            public:

                static void registerKeywords(Keywords& keys);
                explicit DIFF(const ActionOptions& ao);
                virtual void calculate();
        };

        PLUMED_REGISTER_ACTION(DIFF,"DIFF")

    
        // DEFINE KEYWORDS IN plumed.dat INPUT FILE
        void DIFF::registerKeywords(Keywords& keys) {
            Colvar::registerKeywords(keys);
            
            keys.add("compulsory","NAMES_FILE","Atomic numbers");
            
            keys.add("compulsory","PATTERN_FILE","Diffraction pattern");

            keys.add("compulsory","CELL_A","Reference cell A length (Angstrom)");
            keys.add("compulsory","CELL_B","Reference cell B length (Angstrom)");
            keys.add("compulsory","CELL_C","Reference cell C length (Angstrom)");
            keys.add("compulsory","CELL_ALPHA","Reference cell ALPHA angle (Degrees)");
            keys.add("compulsory","CELL_BETA","Reference cell BETA angle (Degrees)");
            keys.add("compulsory","CELL_GAMMA","Reference cell GAMMA angle (Degrees)");
			
			keys.addFlag("DO_GRAD",false,"Calculate gradient");
			
			keys.addFlag("DO_BIAS",false,"Perform biasing algorithm");
			keys.add("optional","GAP","Number of frames between force updates");
			keys.add("optional","FORCE_COEFF","Force coefficient");
            keys.add("optional","FORCE_TYPE","Specifies the type of MD forces calculation (SINGULAR, MEDIAN_INTEGRAL, MEDIAN_POINT, MAX)");
        }
        
        DIFF::DIFF(const ActionOptions& ao): PLUMED_COLVAR_INIT(ao), gap(1)
        {
            using std::string;
            using std::to_string;
            using std::stringstream;
            
			if(torch::cuda::is_available()) {
				log.printf(" DIFF: CUDA is available \n");
			}
			
            std::ifstream in;
			
            int natoms = 0;
            string names, line;
            parse("NAMES_FILE", names);
            in.open(names);
            if(in.is_open()) {
                while(getline(in, line)) {
                    stringstream ss(line);
                    int tempName;
                    ss >> tempName;
                    atomsNames.push_back(tempName - 1);
                    natoms += 1;
                }
                in.close();
            } else { 
                error("Atomic numbers file " + names + " not found"); 
            }
			
			
			string pattern;
			parse("PATTERN_FILE", pattern);
			in.open(pattern);
			
			if(in.is_open()) {
				while(getline(in, line)) {
					stringstream ss(line);
					double tempAngle, tempIntensity;
					ss >> tempAngle >> tempIntensity;
					refPeaksAngles.push_back(tempAngle);
					refPeaksIntensities.push_back(tempIntensity);
				}
				in.close();
			} else { 
				error("Pattern file not found");
			}
			
			parseFlag("DO_GRAD", doGrad);
			parseFlag("DO_BIAS", doBias);
			
			if(doBias) {
				log.printf(" DIFF: biasing is active \n");
				if(!doGrad) {
					log.printf(" DIFF: ignoring DO_GRAD=false flag \n");
					doGrad = true;
				}
				string inForceType;
				parse("FORCE_TYPE", inForceType);
				forceType = ForceType::SINGULAR;
				if(inForceType == "SINGULAR") { forceType = ForceType::SINGULAR; }
				else if (inForceType == "MAX") { forceType = ForceType::MAX; }
				else if (inForceType == "MEDIAN_POINT") { forceType = ForceType::MEDIAN_POINT; }
				else if (inForceType == "MEDIAN_INTEGRAL") { forceType = ForceType::MEDIAN_INTEGRAL; }
				else { error("Unknown force calculation type " + inForceType); }
			
				parse("FORCE_COEFF", forceCoeff);
				if(forceCoeff < 0) {
					log.printf(" DIFF: Pulling mode is active \n"); 
				} else if(forceCoeff == 0) { 
					error("FORCE_COEFF value cannot be zero \n"); 
				} else { 
					error("Pushing mode is not implemented yet \n"); 
				}
			}
			
			auto maxIntensity = std::max_element(refPeaksIntensities.begin(), refPeaksIntensities.end());
			int npts = static_cast<int>(refPeaksAngles.size());
			
			for (int i = 0; i < npts; ++i) {
				refPeaksIntensities[i] /= *maxIntensity;
				refPeaksIntensities[i] *= 100;
			}

			refPeaksIntensitiesTensor = getTensorFromVector(refPeaksIntensities);
			
			double a,b,c,alpha,beta,gamma;
			
			parse("CELL_A", a);
			parse("CELL_B", b);
			parse("CELL_C", c);
			parse("CELL_ALPHA", alpha);
			parse("CELL_BETA", beta);
			parse("CELL_GAMMA", gamma);
			
			if(a <= 0 || b <= 0 || c <= 0 || alpha <= 0 || beta <= 0 || gamma <= 0) {
				error("Reference cell parameters must be positive");
			}
			alpha *= DEG_TO_RAD;
			beta *= DEG_TO_RAD;
			gamma *= DEG_TO_RAD;
			
			parse("GAP", gap);
            if(gap <= 0) {
                error("GAP value cannot be negative or zero");
            }
			
			addValueWithDerivatives();
            setNotPeriodic();
			
            vector<AtomNumber> atoms;
            for(int i = 1; i < natoms+1; ++i){
                AtomNumber d;
                d.setSerial(i);
                atoms.push_back(d);
            }
			requestAtoms(atoms);
			derivatives = vector<Vector>(atoms.size(), Vector(0.0, 0.0, 0.0));

			constants.th2ini = refPeaksAngles.front() * DEG_TO_RAD;
			constants.th2end = refPeaksAngles.back() * DEG_TO_RAD;
			constants.npts = npts;
			constants.initStep();
			
			std::cout.tie(nullptr);
			
			refLattice = Lattice::fromParameters(a, b, c, alpha, beta, gamma);
			refLattice.calc_f_and_theta(atomsNames, constants);
			
			log.printf(" DIFF: initialized \n");

            checkRead();
        }
		
        
        void DIFF::calculate(){
			if(getStep() % gap == 0) {
				torch::Tensor atomsPositions = getTensorFromMatrix(getPositions()) * 10;
				
				atomsPositions = torch::matmul(atomsPositions, refLattice.converter);
				atomsPositions.requires_grad_();
				
				Structure currentStructure(atomsPositions, atomsNames, refLattice);
				Powder currentPowder(currentStructure);
				PreProcess pp(currentPowder.peaksIntensities, torch::masked_select(refLattice.thetas, currentPowder.mask), constants);
				Correlation correlation(pp.newPattern, refPeaksIntensitiesTensor, constants);  
				
				setValue(correlation.dfg.item<double>());
				setBoxDerivativesNoPbc();
				
				if(doGrad) {
					correlation.dfg.backward();
					for(size_t i = 0; i < atomsPositions.sizes()[0]; ++i) {
						for(int j = 0; j < 3; ++j) {
							derivatives[i][j] = atomsPositions.grad()[i][j].item<double>();
						}
					}
				}
			}
			if(doGrad && !doBias) {
				for(size_t i = 0; i < derivatives.size(); ++i) {
					setAtomsDerivatives(i, derivatives[i]);
				}
			}
			else if(doBias) {
				vector<double> mdf(derivatives.size(), 0.0);
				vector<Vector> mdforces(derivatives.size(), Vector(0.0, 0.0, 0.0));
				PLMD::ActionAtomistic::atoms.getLocalMDForces(mdforces); 
				
				switch(forceType) {
					case ForceType::SINGULAR:
					{
						for(size_t i = 0; i < mdforces.size(); ++i) {
							mdf[i] = mdforces[i].modulo();
						}
						break;
					}
					case ForceType::MAX:
					{
						auto maxmdf = std::max_element(mdforces.begin(), mdforces.end(), [](Vector va, Vector vb) { return va.modulo() < vb.modulo(); });
						double maxval = (*maxmdf).modulo();
						for(size_t i = 0; i < mdforces.size(); ++i) {
							mdf[i] = maxval;
						}
						break;
					}
					case ForceType::MEDIAN_POINT:
					{
						std::sort(mdforces.begin(), mdforces.end(), [](Vector va, Vector vb) { return va.modulo() < vb.modulo(); });
						double medianval = mdforces[mdforces.size() / 2].modulo();
						for(size_t i = 0; i < mdforces.size(); ++i) {
							mdf[i] = medianval;
						}
						break;
					}
					case ForceType::MEDIAN_INTEGRAL:
					{
						vector<double> mdmodules;
						double summed = 0.0;
						for(size_t i = 0; i < mdforces.size(); ++i) {
							double tempModulo = mdforces[i].modulo();
							mdmodules.push_back(tempModulo);
							summed += tempModulo;
						}
						std::sort(mdmodules.begin(), mdmodules.end());
						double tempSum = 0.0;
						for(size_t i = 0; i < mdmodules.size(); ++i) {
							tempSum += mdmodules[i];
							if(tempSum >= summed / 2.0) { 
								tempSum = mdmodules[i];
								break;
							}
						}
						for(size_t i = 0; i < mdforces.size(); ++i) {
							mdf[i] = tempSum;
						}
						break;
					}
				}
				std::vector<Vector>& f = modifyForces();
				for(size_t i = 0; i < derivatives.size(); ++i) {
					f[i] = mdf[i]*forceCoeff*derivatives[i];
				}
			}
        }
    }
}
