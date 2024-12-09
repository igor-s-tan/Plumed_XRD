#include "DIFFbase.h"


using std::vector;
using std::sqrt;
using std::sin;
using std::cos;

namespace PLMD {
	

    	
    Constants::Constants()
    {
        initStep();
        initSigmas();
    }
    
    void Constants::initStep() 
    {
        step = (th2end-th2ini) / (npts - 1) / DEG_TO_RAD;
		m = static_cast<int>(std::floor(1.0 / step));
		
		
		weights = torch::zeros({npts, npts}, getOptions());
		for(int i = 0; i < npts; ++i) {
			for(int j = 0; j < npts; ++j) {
				if(std::abs(i - j) <= m) {
					weights[i][j] = std::fmax(1.0 - (std::abs(i - j) * step), 0.0);
				}
			}
		}
		weights_mask = weights.gt(0.0);
		
		weights = torch::masked_select(weights, weights_mask);
    }
    
    void Constants::initSigmas() 
    {
        sigma2 = sigma * sigma;
        sigma_d = 1.0 / (sigma2 * 2.0);
    }
    
    
    
    Lattice Lattice::fromVectors(const torch::Tensor& lattice)
    {
        double asq = 0.0, bsq = 0.0, csq = 0.0;
        double dotAB = 0.0, dotAC = 0.0, dotBC = 0.0;
        for(int i = 0; i < 3; ++i) {
            asq += lattice[0][i].item<double>() * lattice[0][i].item<double>();
            bsq += lattice[1][i].item<double>() * lattice[1][i].item<double>();
            csq += lattice[2][i].item<double>() * lattice[2][i].item<double>();
            dotAB += lattice[0][i].item<double>() * lattice[1][i].item<double>();
            dotAC += lattice[0][i].item<double>() * lattice[2][i].item<double>();
            dotBC += lattice[1][i].item<double>() * lattice[2][i].item<double>();
        }
        
        double a = sqrt(asq);
        double b = sqrt(bsq);
        double c = sqrt(csq);

        using std::acos;
        double alpha = acos(dotBC/bsq/csq);
        double beta = acos(dotAC/asq/csq);
        double gamma = acos(dotAB/asq/bsq);
                       
        return Lattice(a, b, c, alpha, beta, gamma, lattice);
    }

    Lattice Lattice::fromParameters(double a, double b, double c, double alpha, double beta, double gamma)
    {
        double cx = c*cos(beta);
        double cy = c*((cos(alpha) - cos(beta)*cos(gamma))/sin(gamma));
		
		double temp_data[9] = {a, 0.0, 0.0, b*cos(gamma), b*sin(gamma), 0.0, cx, cy, sqrt(c*c - cx*cx - cy*cy)};
        torch::Tensor tempTensor = torch::eye(3, getOptions());
		for(int i = 0; i < 9; ++i) {
			tempTensor[i / 3][i % 3] = temp_data[i];
		}
        
        return Lattice(a, b, c, alpha, beta, gamma, tempTensor);
    }
    
    torch::Tensor Lattice::getReciprocalFromLattice(const torch::Tensor& lattice) 
    {
        torch::Tensor reciprocal = (torch::matmul(lattice, torch::transpose(lattice, 0, 1))).inverse();
        reciprocal.operator*=(BOHR_TO_A*BOHR_TO_A);
        return reciprocal;
    }
    
    torch::Tensor Lattice::getDiagFromTensor(const torch::Tensor& reciprocal)
    {
        torch::Tensor diag = torch::zeros(3, getOptions());
        for(int i = 0; i < 3; ++i) {
            diag[i] = torch::sqrt(reciprocal[i][i]);
        }
        return diag;
    }
	
	torch::Tensor Lattice::getConverterFromLattice(const torch::Tensor& lattice)
	{
        return torch::linalg::inv(torch::transpose(lattice, 0, 1));
	}
	
	void Lattice::calc_f_and_theta(const vector<int>& atomsNames, const Constants& constants) 
	{
		using std::fabs;
        
        double tshift = constants.sigma * sqrt(fabs(-2.0 * std::log(constants.iepscont / constants.intmax))) * DEG_TO_RAD;
        double lambdaBohr = constants.lambda / BOHR_TO_A;
        double smax = sin((constants.th2end + tshift) / 2.0);
		
        int hmax = 2 * (int) ceil(2.0 * smax / lambdaBohr / torch::min(diag).item<double>());
		kvecs = torch::zeros({3, static_cast<int>(std::pow(2.0*hmax + 1, 3.0) - 1)}, getOptions());
		int count = 0;
		for(int hcell = 1; hcell <= hmax; ++hcell) {
			for (int h = -hcell; h <= hcell; ++h) {
				for (int k = -hcell; k <= hcell; ++k) {
					for (int l = -hcell; l <= hcell; ++l) {
						if (abs(h) != hcell && abs(k) != hcell && abs(l) != hcell) {continue;}
						kvecs[0][count] = h;
						kvecs[1][count] = k;
						kvecs[2][count] = l;
						++count;
					}
				}
			}
		}
		torch::Tensor dh2 = torch::sum(torch::matmul(reciprocal, kvecs) * kvecs, 0);
		torch::Tensor dh2mask = torch::abs(dh2).lt(smax);
		dh2 = torch::masked_select(dh2, dh2mask);
		kvecs = torch::masked_select(kvecs, dh2mask);
		kvecs = torch::reshape(kvecs, {3, dh2.sizes()[0]});
		
		torch::Tensor dh = torch::sqrt(dh2);
		thetas = 2.0 * torch::asin(0.5 * lambdaBohr * dh);

		
		double left = constants.th2ini - tshift;
		double right = constants.th2end + tshift;
		
		torch::Tensor left_mask = thetas.gt(left);
		thetas = torch::masked_select(thetas, left_mask);
		torch::Tensor right_mask = thetas.lt(right);
		thetas = torch::masked_select(thetas, right_mask);
		
		
		dh2 = torch::masked_select(dh2, left_mask);
		
		kvecs = torch::masked_select(kvecs, left_mask);
		kvecs = torch::reshape(kvecs, {3, dh2.sizes()[0]});
		
		dh2 = torch::masked_select(dh2, right_mask);
		
		kvecs = torch::masked_select(kvecs, right_mask);
		kvecs = torch::reshape(kvecs, {3, dh2.sizes()[0]});
		
		dh = torch::masked_select(dh, left_mask);
		dh = torch::masked_select(dh, right_mask);
		
		torch::Tensor sthlam = dh / BOHR_TO_A / 2.0;
		
		kvecs *= M_2PI;
		
		vector<vector<double>> current_a_factor, current_b_factor;
		vector<double> current_c_factor;
		
		for(size_t i = 0; i < atomsNames.size(); ++i) {
			current_a_factor.push_back(a_factor[atomsNames[i]]);
			current_b_factor.push_back(b_factor[atomsNames[i]]);
			current_c_factor.push_back(c_factor[atomsNames[i]]);
		}
		
		torch::Tensor a_ffac = getTensorFromVectorOfVectors(current_a_factor);
		torch::Tensor b_ffac = getTensorFromVectorOfVectors(current_b_factor);
		torch::Tensor c_ffac = getTensorFromVector(current_c_factor);

		
		ffacs = torch::einsum("kj,kjl->kl", {a_ffac, torch::exp(-1.0 * torch::einsum("ik,j->kij", {torch::transpose(b_ffac, 0, 1), dh2}))});
		ffacs = torch::add(ffacs, torch::reshape(c_ffac, {c_ffac.sizes()[0], 1}));
		ffacs = ffacs * torch::exp(-1.0 * sthlam * sthlam);

		
		lps = (0.75 + 0.25 * torch::cos(2.0 * thetas)) / torch::sin(thetas) / torch::sin(thetas / 2.0);
	}
    
    Powder::Powder(const Structure& structure): structure(structure)
    {
		_generatePeaks();
    }

    void Powder::_generatePeaks() 
    {		
		peaksIntensities = torch::matmul(structure.atoms, structure.lattice.kvecs);
		peaksIntensities = torch::pow(torch::sum(structure.lattice.ffacs * torch::cos(peaksIntensities), 0), 2) + torch::pow(torch::sum(structure.lattice.ffacs * torch::sin(peaksIntensities), 0), 2);
		mask = peaksIntensities.ge(0.00001);
		peaksIntensities = torch::masked_select(peaksIntensities, mask) * torch::masked_select(structure.lattice.lps, mask);
	}

    
    
    PreProcess::PreProcess(const torch::Tensor& pattern
                         , const torch::Tensor& grid
                         , const Constants& constants)
                     : oldPattern(pattern)
                     , oldGrid(grid)
                     , constants(constants) 
    {
        newGrid = torch::arange(constants.npts, getOptions()) * constants.step + constants.th2ini/DEG_TO_RAD;
		newGrid = torch::reshape(oldGrid / DEG_TO_RAD, {-1, 1}) - newGrid;
		newGrid = torch::exp(torch::pow(newGrid, 2) * constants.sigma_d * (-1.0));

		newPattern = torch::matmul(oldPattern, newGrid);
		newPattern = newPattern / torch::max(newPattern) * 100;
	}
    
    
    void PreProcess::normalizePatternToIntegral(torch::Tensor& pattern, double step) 
    {
        pattern = pattern / torch::sqrt((2.0 * torch::dot(pattern.index({torch::indexing::Slice(1, -1)}), pattern.index({torch::indexing::Slice(1, -1)})) \
			+ pattern.index({0}) * pattern.index({0}) + pattern.index({-1}) * pattern.index({-1})) * step/2.0);
    }


    Correlation::Correlation(const torch::Tensor& current
                   , const torch::Tensor& ref
                   , const Constants& constants) 
				: _current(current)
				, _ref(ref)
				, constants(constants)
    {
		PreProcess::normalizePatternToIntegral(_current, constants.step);
		PreProcess::normalizePatternToIntegral(_ref, constants.step);
        _calcCorrelationOverlap();
    }
        
		
    torch::Tensor Correlation::_calcCorrelationFunction(const torch::Tensor& f, const torch::Tensor& g) 
    {
        return constants.step * constants.step * torch::dot(torch::masked_select(torch::outer(f, g), constants.weights_mask), constants.weights);
    }
    
	
    void Correlation::_calcCorrelationOverlap()
    {
        cfg = _calcCorrelationFunction(_current, _ref);
        cgg = _calcCorrelationFunction(_current, _current);
        cff = _calcCorrelationFunction(_ref, _ref);
        dfg = torch::maximum(torch::eye(1, getOptions()) - cfg / torch::sqrt(cff * cgg), torch::zeros(1, getOptions()));
    }
}