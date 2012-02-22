#include "DistributionFunctions.h"

namespace PLMD {

//+PLUMEDOC MODIFIER WITHIN
/**

Calculates the number of colvars that are within a certain range.  To make this quantity continuous it is calculated using:

\f[
S = \sum_i \int_a^b G( s_i, \sigma*(b-a) )
\f]

where \f$G( s_i, \sigma )\f$ is a normalized Gaussian function of width \f$\sigma\f$ centered at the value of the colvar \f$s_i\f$.  \f$a\f$ and \f$b\f$ are
the lower and upper bounds of the range of interest respectively.  The values of \f$a\f$ and \f$b\f$ must be specified using the WITHIN keyword.  If this keyword
has three input numbers then the third is assumed to be the value of \f$\sigma\f$.  You can specify that you want to investigate multiple rangles by using multiple instances 
of the WITHIN keyword (WITHIN1,WITHIN2 etc).  Alternatively, if you want to calculate a discretized distribution function you can use the HISTOGRAM keyword in 
tandem with the RANGE keyword.  HISTOGRAM takes as input the number of bins in your distribution and (optionally) the value of the smearing parameter \f$\sigma\f$.
RANGE takes the upper and lower bound of the histogram.  The interval between the upper and lower bound specified using RANGE is then divided into 
equal-width bins.  

*/
//+ENDPLUMEDOC

within::within( const std::vector<std::string>& parameters ) :
DistributionFunction(parameters)
{ 
  if( parameters.size()==3 ){
     Tools::convert(parameters[0],a);
     Tools::convert(parameters[1],b); 
     Tools::convert(parameters[2],sigma);
  } else if(parameters.size()==2){
     Tools::convert(parameters[0],a);
     Tools::convert(parameters[1],b);
     sigma=0.5;
  } else {
     error("WITHIN keyword takes two or three arguments");
  }
  if(a>=b) error("For WITHIN keyword upper bound is greater than lower bound");
  hist.set(a,b,sigma);
}

std::string within::message(){
  std::ostringstream ostr;
  ostr<<"number of values between "<<a<<" and "<<b<<" The value of the smearing parameter is "<<sigma;
  return ostr.str();
}

double within::compute( const double p, double& df ){
  double f; f=hist.calculate( p , df );
  return f;
}

double within::last_step( const double p, double& df ){
  df=1.0; return p;
}

}
