#include <cmath>
#include <cstdio>
#include <algorithm>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include "dw_rand.h"

#include "CSampleIDWeight.hpp"
#include "CPutGetBin.hpp"

vector<string> glob(const string &pattern); 

string CPutGetBin::GetFileNameForDump() const
{
	stringstream convert;
        convert << id << "." << nDumpFile << "." << suffix << ".record";
        string file_name = filename_prefix + convert.str();
	return file_name; 
}

void CPutGetBin::GetFileNameForFetchConsolidate()
{
	stringstream convert;
        convert.str(string());
        convert << id << ".*.*.record"; // << ".*" // all cluster node; 

        string filename_pattern = filename_prefix + convert.str();
	filename_fetch.clear(); 
	filename_consolidate.clear(); 

        vector<string>filename = glob(filename_pattern); 
	if (!filename.empty())
	{
		for (int i=0; i<(int)filename.size(); i++)
		{
			if (NumberRecord(filename[i]) >= (int)capacity)
				filename_fetch.push_back(filename[i]); 
			else 
				filename_consolidate.push_back(filename[i]); 
		}
	}
	check_file_fetch_consolidate = true; 
}

bool CPutGetBin::Dump(const string &_filename)
{
	string file_name; 
	if (_filename.empty())
		file_name = GetFileNameForDump(); 
	else 
		file_name = _filename; 

	// Real file operation
	fstream oFile(file_name.c_str(), ios::out | ios::binary); 
	if (!oFile)
	{
	        cerr << "Error in dumping samples to " << file_name << endl; 
		return false;
	}
	for (int i=0; i<nPutUsed; i++)
		write(oFile, &(dataPut[i])); 
	oFile.flush(); 
	oFile.close();  

	return true; 
}

bool CPutGetBin::Fetch()
{
	if (filename_fetch.empty() && (int)capacity > nPutUsed)
		return false;

	int nFetchFile = (int)filename_fetch.size(); 
	if ((int)dataGet.size() < capacity)
		dataGet.resize(capacity);  
	
	int select; 
	vector < vector <int > > select_per_file(nFetchFile+1); 
	// First nFetchFile(): files
	// Last: cache
	for (int i=0; i<(int)capacity; i++)
	{
		select=dw_uniform_int(nFetchFile*capacity+nPutUsed); 
		select_per_file[select/capacity].push_back(select%capacity); 
	}
	int counter =0; 
	for (int i=0; i<(int)nFetchFile; i++)	// Read data from file
	{
		if (!select_per_file[i].empty())
		{
			if (!ReadFromOneFile(filename_fetch[i], counter, select_per_file[i]) )
			{
				cerr << "Error in reading data from " << filename_fetch[i] << endl; 
				return false; 
			}
		}
	}
	if (!select_per_file[nFetchFile].empty() ) // Read data from cache
	{
		for (int n=0; n<(int)select_per_file[nFetchFile].size(); n++)
		{
			dataGet[counter] = dataPut[select_per_file[nFetchFile][n]]; 
			counter ++;
		}
	}
	return true; 
}

bool CPutGetBin::ReadFromOneFile(const string &file_name, int &counter, const vector <int> &index) 
{	
	// real file operation
	fstream iFile; 
        iFile.open(file_name.c_str(), ios::in|ios::binary);
        if (!iFile)
        	return false;
	// determine the dimension of CSampleIDWeight
	CSampleIDWeight sample;
        read(iFile, &sample);
        iFile.seekg(0, ios::beg); 
	int size_each_data = sample.GetSize_Data(); 

        for (int n=0; n<(int)index.size(); n++)
        {
        	iFile.seekg(size_each_data*index[n], ios_base::beg);
                read(iFile, &(dataGet[counter]));
                counter ++;
        }
      	iFile.close();
	return true; 
}

int CPutGetBin::NumberRecord(const string &file_name) const
{
	ifstream input_file(file_name.c_str(), ios::binary|ios::in);
        if (!input_file)
                return -1;
        CSampleIDWeight sample;
        read(input_file, &sample);
        input_file.seekg(0,ios::beg);
        input_file.seekg(0,ios::end);

	int lenFile = input_file.tellg();
	int nSample = lenFile/sample.GetSize_Data();
	if (lenFile % nSample)
	{
		cerr << "CPutGetBin::NumberRecord(): number of data is not an integer " << file_name << endl; 
		return -1; 
	}

	input_file.close(); 
	return nSample;  
}

CPutGetBin::CPutGetBin(const string & _id, int _nDumpFile, int _capacity, string _grandPrefix, int _suffix ) : 
check_file_fetch_consolidate(false),
filename_fetch(vector<string>(0)), 
filename_consolidate(vector<string>(0)),  
suffix(_suffix), id(_id),  nDumpFile(_nDumpFile), capacity(_capacity), 
nPutUsed(0), nGetUsed(_capacity), filename_prefix(_grandPrefix), 
dataPut(vector<CSampleIDWeight>(0)), dataGet(vector<CSampleIDWeight>(0)) 
{
}

CPutGetBin::CPutGetBin(const CPutGetBin &right) : 
check_file_fetch_consolidate(false),
filename_fetch(vector<string>(0)), 
filename_consolidate(vector<string>(0)), 
suffix(right.suffix), id(right.id), nDumpFile(right.nDumpFile), capacity(right.capacity),
nPutUsed(right.nPutUsed), nGetUsed(right.nGetUsed), filename_prefix(right.filename_prefix),
dataPut(right.dataPut), dataGet(right.dataGet)
{
}
 
CPutGetBin & CPutGetBin::operator=(const CPutGetBin &right)
{
	check_file_fetch_consolidate = false; 
	filename_fetch.clear(); 
	filename_consolidate.clear(); 
	suffix = right.suffix; 
	id = right.id; 
	nDumpFile = right.nDumpFile; 
	capacity = right.capacity; 
	nPutUsed = right.nPutUsed; 
	nGetUsed = right.nGetUsed; 
	filename_prefix = right.filename_prefix; 
	dataPut = right.dataPut; 
	dataGet = right.dataGet; 
	return *this; 
}

CPutGetBin::~CPutGetBin()
{
	filename_fetch.clear(); 
	filename_consolidate.clear(); 
	dataPut.clear();
	dataGet.clear();
}

int CPutGetBin::GetNumberFileForDump() const
{
        stringstream convert;
        convert.str(string());
        convert << id << ".*." << suffix << ".record";

        string filename_pattern = filename_prefix + convert.str();

        std::vector<string> filename = glob(filename_pattern);
        return (int)filename.size();
}

int CPutGetBin::DepositSample(const CSampleIDWeight &sample)
{
	int index =  nPutUsed;
	if ((int)dataPut.size() <= nPutUsed)
		dataPut.push_back(sample); 
	else
		dataPut[index] = sample; 
	nPutUsed ++; 
	if (nPutUsed == (int)capacity)
	{
		Dump(); 
		nDumpFile ++; 
		nPutUsed = 0; 
	}
	
	return nDumpFile*capacity+nPutUsed; 
}

bool CPutGetBin::DrawSample(CSampleIDWeight &sample)
{
	int index; 
	if (nGetUsed < (int)capacity)	// if data in the memory has not been exhaustively used 
	{
		index = dw_uniform_int(capacity);
                nGetUsed ++;
                sample = dataGet[index];
		return true; 
	}
	else	// replenish and get data
	{
		if (!check_file_fetch_consolidate )
			GetFileNameForFetchConsolidate(); 
		
		if (filename_fetch.empty() && nPutUsed == 0 )
		{
			// cerr << "Error in DrawSample() : files and bins are empty.\n"; 
			return false; 
		}
		else if (filename_fetch.empty() ) // && nPutUsed > 0) 
		/* when data have not been dumped to files,
 		will directly get a data from dataPut
 		*/
		{
			/*
 			nGetUsed is initialized as capacity and is reset to 0 after each Fetch. 
			This part is executated when there is no files for fetch (and therefore 
			nGetUsed cannot be 0). So we let nGetUsed continue to increase until the 
			net increment, nGetUsed - capacity exceeds the number of samples in 
			nPutUsed.
 			*/
			/*if (nGetUsed >= capacity + nPutUsed)
			{
				cerr << "Error in DrawSample(): files are empty while samples in bins have been exhausitively used.\n"; 
				return false; 
			}*/

			/*
 			If DrawSample is attempted even after all data in dataPut have been exhausted,
			we will let dataPut continue to be used
 			*/
			index = dw_uniform_int(nPutUsed); 
			sample = dataPut[index];
			nGetUsed ++;  
			return true; 
		}
		else	// nFetchFile > 0 and/or nPutUsed > 0, get data from files
		{
			Fetch();
                        nGetUsed = 0;
			index = dw_uniform_int(capacity);
                	nGetUsed ++;
                	sample = dataGet[index];
                	return true;
		}
	}
}

void CPutGetBin::finalize()
{
	if (nPutUsed > 0)
	{
		Dump();
		nDumpFile++;
		nPutUsed =0; 
	}  
}

void CPutGetBin::consolidate()
{
	if (!check_file_fetch_consolidate)
		GetFileNameForFetchConsolidate(); 
	if ((int)filename_consolidate.size() > 1)
	{
		vector <CSampleIDWeight> sample_consolidate; 
		vector <CSampleIDWeight> temp_sample; 
		for (int i=0; i<(int)filename_consolidate.size(); i++)
		{
			temp_sample = LoadSampleFromFile(filename_consolidate[i]); 
			sample_consolidate.insert(sample_consolidate.end(), temp_sample.begin(), temp_sample.end()); 
			remove(filename_consolidate[i].c_str()); 
		}
		int nComplete = (int)sample_consolidate.size()/capacity; 
		int nRemaining = (int)sample_consolidate.size()%capacity; 
		dataPut.resize(capacity); 
		for (int iComplete=0; iComplete<(int)nComplete; iComplete ++)
		{
			copy(sample_consolidate.begin()+iComplete*capacity, sample_consolidate.begin()+(iComplete+1)*capacity, dataPut.begin()); 
			//for (int j=0; j<capacity; j++)
			//	dataPut[j] = sample_consolidate[j+iComplete*capacity]; 
			nPutUsed = capacity; 
			Dump(filename_consolidate[iComplete]); 
		}
		if (nRemaining > 0)
		{
			if ((int)dataPut.size() < nRemaining)
				dataPut.resize(nRemaining); 
			copy(sample_consolidate.begin()+nComplete*capacity, sample_consolidate.begin()+nComplete*capacity+nRemaining, dataPut.begin()); 
			//for (int j=0; j<nRemaining; j++)
			//	dataPut[j] = sample_consolidate[j+nComplete*capacity]; 
			nPutUsed = nRemaining; 
			Dump(filename_consolidate[nComplete]); 
		}
		check_file_fetch_consolidate = false; // because Fetch and Consolidate files changed after consolidate
	}
}

void CPutGetBin::restore()
{
	nDumpFile = GetNumberFileForDump(); 
	if (nDumpFile > 0)
	{
       	 	string file_name;
        	stringstream convert;

        	convert.str(string());
        	convert << id << "." << nDumpFile-1 << "." << suffix << ".record"; 
         	file_name = filename_prefix + convert.str();
		dataPut = LoadSampleFromFile(file_name); 
		nPutUsed = (int)dataPut.size(); 
 
		if (nPutUsed >0 && nPutUsed < (int)capacity)
			nDumpFile --; 
		else if (nPutUsed >= (int)capacity)
			nPutUsed =0; 
	}
}

void CPutGetBin::RestoreForFetch()
{
	if (!check_file_fetch_consolidate)
		GetFileNameForFetchConsolidate(); 
	// After consolidation, there should be at most 1 partial file
	if (!filename_consolidate.empty())
	{
		vector<CSampleIDWeight> tempSample = LoadSampleFromFile(filename_consolidate[0]); 
		if (!tempSample.empty())
		{
			nPutUsed = (int)tempSample.size(); 
			if (dataPut.size() < tempSample.size())
				dataPut.resize(capacity);
			copy(tempSample.begin(), tempSample.end(), dataPut.begin());
			//for (int j=0; j<tempSample.size(); j++)
			//	dataPut[j] = tempSample[j]; 
		}
	}
}
 
bool CPutGetBin::empty() const
{
	if (check_file_fetch_consolidate)
	{
		if (filename_fetch.empty() && filename_consolidate.empty())
			return true; 
		else 
			return false; 
	}
	else 
	{
        	stringstream convert;
        	convert.str(string());
        	convert << id << ".*.*.record"; // << ".*" // all cluster node; 

        	string filename_pattern = filename_prefix + convert.str();

		vector<string> filename = glob(filename_pattern); 
        	if (filename.empty())
			return true; 
		else 
			return false; 
	}
}

void CPutGetBin::DisregardHistorySamples()
{
        if(!check_file_fetch_consolidate)
		GetFileNameForFetchConsolidate();
	vector<string> file; 
	file.insert(file.end(), filename_consolidate.begin(), filename_consolidate.end()); 
	file.insert(file.end(), filename_fetch.begin(), filename_fetch.end());

	for (int iFile =0; iFile<(int)file.size(); iFile++)
		remove(file[iFile].c_str());
	check_file_fetch_consolidate = false; 
	filename_fetch.clear(); 
	filename_consolidate.clear(); 
}

void CPutGetBin::ClearDepositDrawHistory()
{
	nDumpFile = 0; 
 	nGetUsed = capacity;
	nPutUsed = 0; 
	check_file_fetch_consolidate = false; 
	filename_fetch.clear(); 
	filename_consolidate.clear(); 
}
