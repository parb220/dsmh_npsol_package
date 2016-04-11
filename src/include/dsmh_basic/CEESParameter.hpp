#ifndef CLASS_EES_PARAMETER_HEADER
#define CLASS_EES_PARAMETER_HEADER

#include <string>
#include <vector>

using namespace std; 

class CSampleIDWeight; 

class CEESParameter 
{
public:
	// The following parameter can either be determined from reading the file or command line or by default values (specified in a header file)
	string storage_dir;	// fold to store all relevant information
	int storage_marker;		// number of samples stored in memory for retrieve 
	string run_id; 
	int number_energy_stage;	// number of energy stages
	int number_striation; 	// number of rings
	double pee; 	// probability of equi-energy-jump
	double lambda_1;	// 1.0/tN_1
	vector<double> lambda; 	// 1.0/t
	
	int highest_stage; 
	int lowest_stage; 
	int THIN;  	// thinning factor for MH samples
	int simulation_length; 
	int burn_in_length; 
public:
	CEESParameter(); 
	~CEESParameter(); 
	
	bool SetTemperature_geometric(); 
	bool SetTemperature_quadratic();  
	bool SetTemperature_polynomial(double r);

	double LogRatio_Stage(const CSampleIDWeight &x, const CSampleIDWeight &y, int stage) const; 
};
#endif
