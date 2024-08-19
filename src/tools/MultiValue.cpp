/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2014-2023 The plumed team
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
#include "MultiValue.h"
#include "Tools.h"

namespace PLMD {

MultiValue::MultiValue( const size_t& nvals, const size_t& nder, const size_t& nmat, const size_t& maxcol ):
  task_index(0),
  task2_index(0),
  values(nvals),
  nderivatives(nder),
  derivatives(nvals*nder),
  hasderiv(nvals*nder,false),
  nactive(nvals),
  active_list(nvals*nder),
  atLeastOneSet(false),
  vector_call(false),
  nindices(0),
  nsplit(0),
  nmatrix_cols(maxcol),
  matrix_force_stash(nder*nmat),
  matrix_row_nderivatives(nmat,0),
  matrix_row_derivative_indices(nmat)
{
  for(unsigned i=0; i<nmat; ++i) matrix_row_derivative_indices[i].resize( nder );
}

void MultiValue::resize( const size_t& nvals, const size_t& nder, const size_t& nmat, const size_t& maxcol ) {
  if( values.size()==nvals && nderivatives==nder && matrix_row_nderivatives.size()==nmat && nmatrix_cols==maxcol ) return;
  values.resize(nvals); nderivatives=nder; derivatives.resize( nvals*nder );
  hasderiv.resize(nvals*nder,false); nactive.resize(nvals); active_list.resize(nvals*nder);
  nmatrix_cols=maxcol; matrix_force_stash.resize(nmat*nder,0);
  matrix_row_nderivatives.resize(nmat,0); matrix_row_derivative_indices.resize(nmat); atLeastOneSet=false;
  for(unsigned i=0; i<nmat; ++i) matrix_row_derivative_indices[i].resize( nder );
}

void MultiValue::clearAll() {
  for(unsigned i=0; i<values.size(); ++i) values[i]=0;
  // Clear matrix derivative indices
  for(unsigned i=0; i<matrix_row_nderivatives.size(); ++i) {
      unsigned base = i*nderivatives;
      for(unsigned j=0; j<matrix_row_nderivatives[i]; ++j) matrix_force_stash[base + matrix_row_derivative_indices[i][j]] = 0;
      matrix_row_nderivatives[i]=0;
  }
#ifndef NDEBUG
  for(unsigned i=0; i<matrix_force_stash.size(); ++i) plumed_assert( fabs(matrix_force_stash[i])<epsilon );
#endif
  if( !atLeastOneSet ) return;
  for(unsigned i=0; i<values.size(); ++i) clearDerivatives(i);
  atLeastOneSet=false;
}

void MultiValue::clearDerivatives( const unsigned& ival ) {
  values[ival]=0;
  if( !atLeastOneSet ) return;
  unsigned base=ival*nderivatives;
  for(unsigned i=0; i<nactive[ival]; ++i) {
    unsigned k = base+active_list[base+i]; derivatives[k]=0.; hasderiv[k]=false;
  }
  nactive[ival]=0;
#ifndef NDEBUG
  for(unsigned i=0; i<nderivatives; ++i) {
    if( hasderiv[base+i] ) {
      std::string num1, num2;
      Tools::convert(ival,num1); Tools::convert(i,num2);
      plumed_merror("FAILING TO CLEAR VALUE " + num1 + " DERIVATIVE " + num2 + " IS PROBLEMATIC");
    }
    // As this is debugging code we hard code an escape here otherwise this check is really expensive
    if( i>1000 ) return;
  }
#endif
}

}
