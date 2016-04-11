#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <errno.h>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "CSampleIDWeight.hpp"
#include "CStorageHead.hpp"
#include "CPutGetBin.hpp"

using namespace std;

vector<string> glob(const string &pattern); 

CStorageHead::CStorageHead(int _node_index, const string & _run_id, int _storage_marker, string _file_location, int _number_stage) : 
cluster_node(_node_index), 
run_id(_run_id), 
storage_marker(_storage_marker), 
filename_base(_file_location), 
bin(vector<vector<CPutGetBin> >(_number_stage+1)), 
energy_lower_bound(vector<vector<double> >(_number_stage+1)) 
{
}

void CStorageHead::InitializeBin(int stage)
{
	if (stage < 0 || stage >= (int)bin.size())
	{
		cerr << "CStorageHead::InitializeBin() : stage exceeds range"; 
		abort(); 
	}
	bin[stage].clear(); 

	stringstream str; 
	str << run_id << "/" << run_id << ".binary/";

	// bin[i]: bins for the i-th temperature stage 
	// bin[i][j]: j-th bin for the i-th temperature stage 
	// energy_lower_bound[i]: energy lower bounds for the i-th temperature stage 
	// energy_lower_bound[i][j]: energy lower bound for the j-th bin of the i-th stage 
	
	stringstream bin_id_string; 
	// Each stage has one bin at the begging for depositing
	bin_id_string << stage << ".0"; 
	bin[stage].push_back(CPutGetBin(bin_id_string.str(),0,storage_marker,filename_base+str.str(), cluster_node)); 
}

CStorageHead::~CStorageHead()
{
}

bool CStorageHead::makedir()
{
	stringstream str; 
	str.str(string()); 
	str << run_id; 
	string dir = filename_base + str.str(); 
	errno = 0; // clear error number
	int status = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
	if (status !=0 && errno != EEXIST) { cerr << dir << endl;
	  return false;}
 
	str.str(string()); 
	str << run_id << "/" << run_id << ".binary" ; 
	dir = filename_base + str.str();
	errno = 0; 
	status = mkdir(dir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); 
	if (status == 0 || errno == EEXIST)
		return true; 
	else { cerr << dir << endl;
	  return false; }
}

int CStorageHead::BinIndex(int stage, double energy) const 
{
	if (stage < 0 || stage >= (int)energy_lower_bound.size())
	{
		cerr << "CStorageHead::BinIndex() : stage index exceeds the range.\n"; 
		abort(); 
	}
	if (energy_lower_bound[stage].empty()) // binning has not been done yet
		return 0; 
	for (int j=1; j<(int)energy_lower_bound[stage].size(); j++)
	{
		if (energy < energy_lower_bound[stage][j])
			return j-1; 
	}
	return energy_lower_bound[stage].size()-1; 
}

int CStorageHead::DepositSample(int stage, int _bin_id, const CSampleIDWeight &sample)
{
	if (stage < 0 || stage >= (int)bin.size())
	{
		cerr << "CStorageHead::DepositSample() : stage index exceeds the range.\n"; 
		abort(); 
	}
	if (bin[stage].empty())
	{
		cerr << "CStorageHead::DepositSample() : there should be at least one bin for the stage"; 
		abort(); 
	}
	return bin[stage][_bin_id].DepositSample(sample); 
}

void CStorageHead::finalize(int stage)
{
	if (stage < 0 || stage >= (int)bin.size())
	{
		cerr << "CStorageHead::finalize() : stage index exceeds the range.\n"; 
		abort(); 
	}
	for (int j=0; j<(int)bin[stage].size(); j++)
		bin[stage][j].finalize(); 
}

void CStorageHead::consolidate(int stage)
{
	if (stage < 0 || stage >=(int)bin.size())
	{
		cerr << "CStorageHead::consolidate() : stage index exceeds the range.\n"; 
		abort(); 
	}
	for (int j=0; j<(int)bin[stage].size(); j++)
		bin[stage][j].consolidate(); 
}

void CStorageHead::restore(int stage)
{
	if (stage < 0 || stage >= (int)bin.size())
	{
		cerr << "CStorageHead::restore() : stage index exceeds the range.\n"; 
                abort();
	}
	for (int j=0; j<(int)bin[stage].size(); j++)
		bin[stage][j].restore(); 
}

void CStorageHead::ClearDepositDrawHistory(int stage)
{
	if (stage < 0 || stage >= (int)bin.size())
	{
		cerr << "CStorageHead::ClearDepositDrawHistory() : stage index exceeds the range.\n"; 
		abort(); 
	}
	for (int j=0; j<(int)bin[stage].size(); j++) 
		bin[stage][j].ClearDepositDrawHistory(); 
}

void CStorageHead::ClearSample(int stage)
{
	if (stage <0 || stage >= (int)bin.size())
	{
		cerr << "CStorageHead::ClearSample() : stage index exceeds the range.\n"; 
		abort(); 
	}
	for (int j=0; j<(int)bin[stage].size(); j++)
		bin[stage][j].DisregardHistorySamples(); 
}


std::vector<CSampleIDWeight> CStorageHead::binning_equal_size(int stage, int bin_number, double lambda)
{
	if (stage < 0 || stage >=(int)bin.size())
        {
                cerr << "CStorageHead::binning_equal_size() : stage index exceeds the range.\n";
                abort();
        }
	if (bin[stage].size() > 1)
        {
                cerr << "CStorageHead::binning_equal_size() : it seems that binning has been done before.\n";
                abort();
        }
	ClearStatus(stage);
        consolidate(stage);
        std::vector<CSampleIDWeight> sample = DrawAllSample(stage); 
        if (sample.empty() )
        {
                cerr << "CStorageHead::binning_equal_size() : error occurred when loading all samples.\n";
                abort();
        }
	CSampleIDWeight_Sorter comparator(lambda); 
	sort(sample.begin(), sample.end(), comparator);
	// descending on  lambda*log_likelihood + log_prior
        DisregardHistorySamples(stage);

        stringstream str, bin_id_string;
        str << run_id << "/" << run_id << ".binary/";
        bin[stage].clear();
	
	int sBin = (int)ceil((double)(sample.size())/(double)bin_number);
	
	int iSample=0, iBin=0;
	while (iSample < (int)(sample.size())) 
	{
		// energy_lower_bound[stage].push_back(-sample[iSample].weight); 	
		energy_lower_bound[stage].push_back(-(sample[iSample].reserved*lambda + (sample[iSample].weight-sample[iSample].reserved))); 
		bin_id_string.str(string());
                bin_id_string << stage << "." << iBin;
                bin[stage].push_back(CPutGetBin(bin_id_string.str(),0,storage_marker,filename_base+str.str(), cluster_node));

		sBin = iSample+sBin == (int)(sample.size())-1 ? sBin+1 : sBin;
                for (int j=iSample; j<iSample+(int)sBin && j<(int)(sample.size()); j++)
                	bin[stage][iBin].DepositSample(sample[j]);
               	iSample += sBin;
               	iBin++;
	}
	finalize(stage); 
	ClearDepositDrawHistory(stage); 
	return sample; 
}

bool CStorageHead::DrawSample(int stage, int _bin_id, CSampleIDWeight &sample)
{
	if (stage < 0 || stage >= (int)bin.size() || _bin_id < 0 || _bin_id >= (int)bin[stage].size() )
        {
                cerr << "CStorageHead::DrawSample(): stage or bin index exceeds the range.\n";
                abort();
        }

	return bin[stage][_bin_id].DrawSample(sample); 	
}

std::vector<CSampleIDWeight>  CStorageHead::DrawAllSample(int stage)
{
	if (stage <0 || stage >= (int)bin.size())
	{
		cerr << "CStorageHead::DrawAllSample : stage index exceeds the range.\n"; 
		abort(); 
	}
	stringstream convert; 
       	convert << run_id << "/" << run_id << ".binary/" << stage << ".*.record"; 
	string filename_pattern = filename_base+convert.str(); 
	
       	vector<string> filename = glob(filename_pattern); 
	vector<CSampleIDWeight> sample(0);
       	for (int i=0; i<(int)filename.size(); i++)
       	{
		std::vector<CSampleIDWeight> sample_block = LoadSampleFromFile(filename[i]); 
		sample.insert(sample.end(), sample_block.begin(), sample_block.end()); 
	}
	return sample; 
}

void CStorageHead::DisregardHistorySamples(int stage)
{
	if (stage <0 || stage >= (int)bin.size())
	{
		cerr << "CStorageHead::DisregardHistorySamples(): stage index exceeds the range.\n"; 
		abort(); 
	}
	for (int j=0; j<(int)bin[stage].size(); j++)
		bin[stage][j].DisregardHistorySamples();
}

bool CStorageHead::empty(int stage) const
{
	if (stage<0 || stage >= (int)bin.size() )
	{
		cerr << "CStorageHead::empty(): stage index exceeds the range.\n"; 
		abort(); 
	}
	for (int j=0; j<(int)bin[stage].size(); j++)
	{
		if (!bin[stage][j].empty())
			return false; 
	}
	return true; 
}

void CStorageHead::RestoreForFetch(int stage)
{
	if (stage <0 || stage >=(int) bin.size() )
	{
		cerr << "CStorageHead::RestoreForFetch(): stage index exceeds the range.\n";
		abort(); 
	}
	for (int j=0; j<(int)bin[stage].size(); j++)
		bin[stage][j].RestoreForFetch(); 
}

void CStorageHead::ClearStatus(int stage)
{
	if (stage <0 || stage >= (int)bin.size())
	{
		cerr << "CStorageHead::ClearStatus(): stage index exceeds the range.\n"; 
		abort(); 
	}
	for (int iBin=0; iBin<(int)(bin[stage].size()); iBin++)
		bin[stage][iBin].ClearStatus(); 
}


int CStorageHead::GetNumber_Bin(int stage) const 
{ 
	if (stage <0 || stage >= (int)energy_lower_bound.size())
        {
                cerr << "CStorageHead::GetNumber_Bin() : stage index exceeds the range.\n";
                abort();
        }

	return (int)energy_lower_bound[stage].size(); 
}

double CStorageHead::GetEnergyLowerBound(int stage, int index) const 
{ 	
	if (stage <0 || stage >= (int)energy_lower_bound.size() || index < 0 || index >= (int)(energy_lower_bound[stage].size()))
        {
                cerr << "CStorageHead::GetEnergyLowerBound() : stage or bin index exceeds the range.\n";
                abort();
        }

	return energy_lower_bound[stage][index]; 
}

void CStorageHead::ResizeBin(int stage, int number)
{
	if (stage <0 || stage >= (int)bin.size() || number < 0 )
        {
                cerr << "CStorageHead::ResizeBin() : stage index exceeds the range or invalid number of bins.\n";
                abort();
        } 
	if (bin[stage].empty())
		InitializeBin(stage); 
	bin[stage].resize(number, bin[stage][0]); 
	stringstream convert; 
	for (int i=0; i<number; i++)
	{
		convert.str(string()); 
		convert << stage << "." << i; 
		bin[stage][i].SetBinID(convert.str(), 0); 
	}
	energy_lower_bound[stage].resize(number); 
}

void CStorageHead::SetEnergyLowerBound(int stage, int index, double e) 
{ 
	if (stage <0 || stage >= (int)energy_lower_bound.size() || index < 0 || index >= (int)(energy_lower_bound[stage].size()))
        {
                cerr << "CStorageHead::SetEnergyLowerBound() : stage or bin index exceeds the range.\n";
                abort();
        }
	energy_lower_bound[stage][index] = e; 
}

void CStorageHead::ClearBin(int stage)
{
	if (stage <0 || stage >= (int)bin.size())
        {
                cerr << "CStorageHead::ClearBin() : stage index exceeds the range.\n";
                abort();
        }
	bin[stage].clear(); 
	bin[stage].swap(bin[stage]); 
}
