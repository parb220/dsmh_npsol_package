#include <cmath>
#include <algorithm>
#include "dw_rand.h"
#include "dw_math.h"
#include "CEquiEnergyModel.hpp"

vector<double> CEquiEnergyModel::Reweight(const vector<CSampleIDWeight> &samples, int current_stage, int previous_stage) 
{
	vector<double> log_weight(samples.size(),0.0);
  	double w = parameter->lambda[current_stage] - parameter->lambda[previous_stage];
  	for (unsigned int i=0; i < samples.size(); i++) 
		log_weight[i]=w*samples[i].reserved;
  	return log_weight;
}

std::vector<CSampleIDWeight> CEquiEnergyModel::Initialize_WeightedSampling(const std::vector<CSampleIDWeight> &samples, int K, int stage_index)
{
	if (!samples.empty())
	{
		std::vector<CSampleIDWeight> starters(K); 
		// Cumulative sum of importance weights
		vector<double> log_weight = Reweight(samples, stage_index-1, stage_index); 
		vector<double> log_weight_sum(samples.size()), weight_sum(samples.size()); 
		log_weight_sum[0] = log_weight[0];  
		for (int i=1; i<(int)samples.size(); i++)
			log_weight_sum[i] = AddLogs(log_weight_sum[i-1], log_weight[i]); 
		for (int i=0; i<(int)samples.size(); i++) // Normalize
			weight_sum[i] = exp(log_weight_sum[i] -log_weight_sum.back()); 

		for (int i=0; i<K; i++)
		{
			double random_number = dw_uniform_rnd(); 
			int position = std::lower_bound(weight_sum.begin(), weight_sum.end(), random_number)-weight_sum.begin(); 
			starters[i] = samples[position]; 	
		} 
		return starters; 
	}
	else 
		return std::vector<CSampleIDWeight>(0); 
}

std::vector<CSampleIDWeight> CEquiEnergyModel::Initialize_WeightedSampling(const std::vector<CSampleIDWeight> &samples, int K)
{
	if (!samples.empty())
	{
		std::vector<CSampleIDWeight> starters(K); 
		// Cumulative sum of importance weights
		vector<double> log_weight_sum(samples.size()), weight_sum(samples.size()); 
		log_weight_sum[0] = samples[0].weight;  
		for (int i=1; i<(int)samples.size(); i++)
			log_weight_sum[i] = AddLogs(log_weight_sum[i-1], samples[i].weight); 
		for (int i=0; i<(int)samples.size(); i++) // Normalize
			weight_sum[i] = exp(log_weight_sum[i] -log_weight_sum.back()); 

		for (int i=0; i<K; i++)
		{
			double random_number = dw_uniform_rnd(); 
			int position = std::lower_bound(weight_sum.begin(), weight_sum.end(), random_number)-weight_sum.begin(); 
			starters[i] = samples[position]; 	
		} 
		return starters; 
	}
	else 
		return std::vector<CSampleIDWeight>(0); 
}
