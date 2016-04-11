#ifndef CLASS_SAMPLE_ID_WEIGHT_HEADER
#define CLASS_SAMPLE_ID_WEIGHT_HEADER

#include <iostream>
#include <fstream>
#include <vector>
#include "dw_dense_matrix.hpp"

using namespace std; 

class CEquiEnergyModel; 

class CSampleIDWeight
{
public:
	TDenseVector data; 
	int id; 
	double weight;
	double reserved;  
	bool calculated; 

	CSampleIDWeight(const TDenseVector &_x=TDenseVector(), int _id=0, double _weight=0.0, double _reserved=0.0, bool _calculated=false); 
	CSampleIDWeight(const CSampleIDWeight &); 
	~CSampleIDWeight();

	CSampleIDWeight & operator=(const CSampleIDWeight &);
	bool PartialCopyFrom(const CSampleIDWeight &, int offset, int length);
	bool PartialCopyFrom(int offset1, const CSampleIDWeight &, int offset2, int length);

	int GetSize_Data() const; 
	void DataChanged(); 

	friend istream& read(istream &, CSampleIDWeight *); 
	friend ostream& write(ostream &, const CSampleIDWeight *);

	friend istream& operator>>(istream &, CSampleIDWeight &); 
	friend ostream& operator<<(ostream &, const CSampleIDWeight &); 
}; 

bool compare_CSampleIDWeight(const CSampleIDWeight &i, const CSampleIDWeight &j);
bool compare_CSampleIDWeight_BasedOnEnergy(const CSampleIDWeight &i, const CSampleIDWeight &j);
bool compare_CSampleIDWeight_BasedOnID(const CSampleIDWeight &i, const CSampleIDWeight &j);

class CSampleIDWeight_Sorter
{
private:
	double lambda; 
public:
	CSampleIDWeight_Sorter(double _lambda = 1.0); 
	bool operator() (const CSampleIDWeight &left, const CSampleIDWeight &right); 
}; 

std::vector<CSampleIDWeight> LoadSampleFromFile(const string &file); 
bool SaveSampleToFile(const string &file, const vector<CSampleIDWeight> &Y); 

#endif
