#ifndef __PLUMED_colvar_DIFFbase_h
#define __PLUMED_colvar_DIFFbase_h

#include "DIFFconstants.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>


using std::vector;

namespace PLMD {
    
	enum class ForceType {MAX, MEDIAN_POINT, MEDIAN_INTEGRAL, SINGULAR};
	
    class Constants
    {   
        public:
			void initStep();
			void initSigmas();
			
			double lambda = 1.5406;
			
            double th2ini = 5.0 * DEG_TO_RAD;
            double th2end = 50.0 * DEG_TO_RAD;
			double window_size = 6.0;
			
            int npts = 10001;
			
			double sigma = 0.05;
            double sigma2, sigma_d;
            
            double step, m;
			
			torch::Tensor weights, weights_mask;
			
			double ieps = 1e-5;
			double iepscont = 1e-10;
			double intmax = 1e15;
			
            explicit Constants();
        
    };
    
    class Lattice
    {
		explicit Lattice(double a, double b, double c
						, double alpha, double beta, double gamma
						, const torch::Tensor& lattice)
					: a(a), b(b), c(c)
					, alpha(alpha), beta(beta), gamma(gamma)
					, lattice(lattice)
					, reciprocal(getReciprocalFromLattice(lattice))
					, diag(getDiagFromTensor(reciprocal))
					, converter(getConverterFromLattice(lattice)) {}
					
        
        public:
        
            explicit Lattice() {}
						
            double a, b, c, alpha, beta, gamma;
			
            torch::Tensor lattice, reciprocal, converter;
            torch::Tensor diag;
			
			torch::Tensor ffacs, thetas, lps, kvecs;
            
			void calc_f_and_theta(const vector<int>& atomsNames
								, const Constants& constants);
			
            static torch::Tensor getReciprocalFromLattice(const torch::Tensor& lattice);
            static torch::Tensor getDiagFromTensor(const torch::Tensor& reciprocal);
            static torch::Tensor getConverterFromLattice(const torch::Tensor& lattice);
            
            static Lattice fromVectors(const torch::Tensor& lattice);
            static Lattice fromParameters(double a, double b, double c
                                        , double alpha, double beta, double gamma);
    };
    
    class Structure
    {
        public:
            explicit Structure(const torch::Tensor& atoms
                             , const vector<int>& atomsNames
                             , const Lattice& lattice)
                         : atoms(atoms)
                         , atomsNames(atomsNames)
                         , lattice(lattice) {}
	
			const torch::Tensor& atoms;
            const vector<int>& atomsNames;
            const Lattice& lattice;
    };
    
    class Powder
    {
		void _generatePeaks();            
		
        public:
            
            explicit Powder(const Structure& structure);
			
			const Structure& structure;
			
			torch::Tensor mask;
			torch::Tensor peaksIntensities;
			torch::Tensor peaksAngles;
    };
    
    
    class PreProcess
    {
        public:
        
            explicit PreProcess(const torch::Tensor& pattern
                              , const torch::Tensor& grid
                              , const Constants& constants);
                    
            static void normalizePatternToIntegral(torch::Tensor& pattern, double step);
            
			torch::Tensor newPattern;
			torch::Tensor newGrid;
			
			const torch::Tensor& oldPattern;
			const torch::Tensor& oldGrid;
			const Constants& constants;
            
    };



    class Correlation
    {
		torch::Tensor _calcCorrelationFunction(const torch::Tensor& f, const torch::Tensor& g);
		void _calcCorrelationOverlap();
		
        public:
		
            explicit Correlation(const torch::Tensor& current
                               , const torch::Tensor& ref
                               , const Constants& constants);
                           
			torch::Tensor dfg, cfg, cgg, cff, _ref, _current;
			const Constants& constants;
    };
}

#endif