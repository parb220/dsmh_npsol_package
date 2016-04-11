#ifndef CLASS_STORAGE_HEAD_HEADER
#define CLASS_STORAGE_HEAD_HEADER

#include <vector>
#include <string>

using namespace std; 

class CSampleIDWeight; 
class CPutGetBin; 

class CStorageHead
{
protected:
	int cluster_node;
	string run_id; 
	int storage_marker; 
	string filename_base; 
	
	vector<vector<CPutGetBin> > bin;
	vector<vector<double> > energy_lower_bound; 
public: 
	CStorageHead(int _node_index=0, const string & _run_id=string(), int _storage_marker=10000, string _file_location=string(), int _number_stage=1); 
	~CStorageHead(); 

	bool makedir(); 
	
	int BinIndex(int stage, double energy) const;  
	int DepositSample(int stage, int _bin_id, const CSampleIDWeight &) ;
	void finalize(int stage);
	void consolidate(int stage); 
	void restore(int stage);
	void ClearDepositDrawHistory(int stage); 
	void ClearSample(int stage); 

	bool DrawSample(int stage, int _bin_id, CSampleIDWeight &sample);
	std::vector<CSampleIDWeight> binning_equal_size(int stage, int bin_number, double lambda); 
	std::vector<CSampleIDWeight> DrawAllSample(int stae); 
		
	/* for reassigning samples into different bins */
	void DisregardHistorySamples(int stage); 
	void RestoreForFetch(int stage); 
	bool empty(int stage) const; 
	
	/* reset setting */
	void ClearStatus(int ); 

	/* get set energy lower bound */
 	int GetNumber_Bin(int stage) const;
        double GetEnergyLowerBound(int stage, int index) const;
        void ResizeBin(int stage, int number); 
        void SetEnergyLowerBound(int stage, int index, double e); 
	void ClearBin(int stage); 
	void InitializeBin(int stage); 
}; 

#endif
