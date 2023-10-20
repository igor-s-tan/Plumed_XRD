/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016-2023 The plumed team
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
#include "core/ActionWithArguments.h"
#include "core/ActionPilot.h"
#include "core/ActionRegister.h"
#include "tools/OFile.h"
#include "core/PlumedMain.h"
#include "FindContour.h"

namespace PLMD {
namespace contour {

//+PLUMEDOC GRIDANALYSIS DUMPCONTOUR 
/*

*/
//+ENDPLUMEDOC

class DumpContour : 
public ActionWithArguments,
public ActionPilot {
private:
  std::string fmt, filename;
  std::vector<unsigned> preps;
  bool output_for_all_replicas;
public:
  static void registerKeywords( Keywords& keys );
  explicit DumpContour(const ActionOptions&ao);
  ~DumpContour() {}
  void calculate() override {}
  void apply() override {}
  void update() override ;
};

PLUMED_REGISTER_ACTION(DumpContour,"DUMPCONTOUR")

void DumpContour::registerKeywords( Keywords& keys ) {
  Action::registerKeywords( keys );
  ActionPilot::registerKeywords( keys ); 
  ActionWithArguments::registerKeywords( keys ); keys.use("ARG");
  keys.add("compulsory","STRIDE","1","the frequency with which the grid should be output to the file.");
  keys.add("compulsory","FILE","density","the file on which to write the grid.");
  keys.add("compulsory","REPLICA","0","the replica for which you would like to output this information");
  keys.add("optional","FMT","the format that should be used to output real numbers");
}

DumpContour::DumpContour(const ActionOptions&ao):
  Action(ao),
  ActionWithArguments(ao),
  ActionPilot(ao),
  fmt("%f"),
  output_for_all_replicas(false)
{
  if( getNumberOfArguments()!=1 ) error("should only be one argument");
  FindContour* fc=dynamic_cast<FindContour*>( getPntrToArgument(0)->getPntrToAction() );
  if( !fc ) error("can only use this action to print data from FIND_CONTOUR actions");

  parse("FILE",filename);
  if(filename.length()==0) error("name out output file was not specified");

  log.printf("  outputting contour with label %s to file named %s",getPntrToArgument(0)->getName().c_str(), filename.c_str() );
  parse("FMT",fmt); log.printf(" with format %s \n", fmt.c_str() ); fmt = " " + fmt;

  std::string rep_data; parse("REPLICA",rep_data);
  if( rep_data=="all" ) output_for_all_replicas=true;
  else { preps.resize(1); Tools::convert( rep_data, preps[0] ); }
  if( output_for_all_replicas ) log.printf("  outputting files for all replicas \n");
  else {
    log.printf("  outputting data for replicas ");
    for(unsigned i=0; i<preps.size(); ++i) log.printf("%d ", preps[i] );
  }

  checkRead();
}

void DumpContour::update() {
  if( !output_for_all_replicas ) {
      bool found=false; unsigned myrep=plumed.multi_sim_comm.Get_rank();
      for(unsigned i=0; i<preps.size(); ++i) {
        if( myrep==preps[i] ) { found=true; break; }
      }
      if( !found ) return;
  }
  OFile ofile; ofile.link(*this);
  ofile.setBackupString("analysis"); 
  ofile.open( filename ); 

  FindContour* fc=dynamic_cast<FindContour*>( getPntrToArgument(0)->getPntrToAction() );
  unsigned maxp = fc->active_cells.size(), ncomp = fc->getNumberOfComponents();
  unsigned ntasks = 0; for(unsigned i=0; i<maxp; ++i) ntasks += fc->active_cells[i];

  ofile.printf("%d\n", ntasks ); ofile.printf("Points found on isocontour\n");
  for(unsigned i=0; i<maxp; ++i) {
      if( fc->active_cells[i]==0 ) continue ;
      const char* defname="X"; const char* name=defname; ofile.printf("%s", name);
      for(unsigned j=0; j<ncomp; ++j ) ofile.printf((" " + fmt).c_str(), (fc->copyOutput(j))->get(i)  );
      ofile.printf("\n"); 
  }
}


}
}
