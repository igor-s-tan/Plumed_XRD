/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2012-2014 The plumed team
   (see the PEOPLE file at the root of the distribution for a list of names)

   See http://www.plumed-code.org for more information.

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
#ifndef __PLUMED_dimred_DimensionalityReductionBase_h
#define __PLUMED_dimred_DimensionalityReductionBase_h

#include "analysis/AnalysisBase.h"

namespace PLMD {
namespace dimred {

class DimensionalityReductionBase : public analysis::AnalysisBase {
friend class ProjectNonLandmarkPoints;
friend class SketchMapBase;
private:
/// This are the target distances for a single point. 
/// This is used when we do out of sample or pointwise global optimization
  std::vector<double> dtargets;
/// The projections that were generated by the dimensionality reduction algorithm
  Matrix<double> projections;
protected:
/// Dimensionality of low dimensional space
  unsigned nlow;
/// A pointer to any dimensionality reduction base that we have got projections from
  DimensionalityReductionBase* dimredbase;
public:
  static void registerKeywords( Keywords& keys );
  DimensionalityReductionBase( const ActionOptions& );
/// Get the ith data point (this returns the projection)
  virtual void getProjection( const unsigned& idata, std::vector<double>& point, double& weight );
/// Actually perform the analysis
  virtual void performAnalysis();
/// Overwrite getArguments so we get arguments from underlying class
  std::vector<Value*> getArgumentList();
/// Calculate the projections of points
  virtual void calculateProjections( const Matrix<double>& , Matrix<double>& )=0;
/// Set one of the elements in the target vector.  This target vector is used
/// when we use calculateStress when finding the projections of individual points.
/// For example this function is used in PLMD::dimred::ProjectOutOfSample
  virtual void setTargetDistance( const unsigned& , const double& );
/// Calculate the pointwise stress on one point when it is located at pp.  
/// This function makes use of the distance data in dtargets
/// It is used in PLMD::dimred::ProjectOutOfSample and in pointwise optimisation
  virtual double calculateStress( const std::vector<double>& pp, std::vector<double>& der );
/// Overwrite virtual function in ActionWithVessel
  void performTask( const unsigned& , const unsigned& , MultiValue& ) const { plumed_error(); }
};

inline
void DimensionalityReductionBase::setTargetDistance( const unsigned& idata, const double& dist ){
  dtargets[idata]=dist;
}

}
}
#endif
