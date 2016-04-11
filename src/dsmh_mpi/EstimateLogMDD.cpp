#include <sstream>
#include <string>
#include <iomanip>
#include <algorithm>
#include "CEquiEnergyModel.hpp"
#include "CSampleIDWeight.hpp"
#include "CStorageHead.hpp"
#include "dw_matrix.h"
#include "storage_constant.hpp" 
#include "prcsn.h"
#include "dw_math.h"
#include "mdd_function.h"
#include "mdd_constant.hpp"
#include "dw_elliptical.h"
#include "EstimateLogMDD.hpp"

using namespace std; 

// Function used in LogMDD calculationg (no need for external interface)
static TMatrix CreateProposalMatrix(int ndraws, CEquiEnergyModel &model, double t, const TElliptical *elliptical, int posterior_type=POSTERIOR_HEATED); 

static TMatrix CreatePosteriorMatrix(const vector<CSampleIDWeight> &sample, double t, const TElliptical *elliptical, int posterior_type=POSTERIOR_HEATED);

static TDenseVector GetRadiusFromSample(const vector<CSampleIDWeight> &sample, const TDenseVector &center, const TDenseMatrix &scale); 

static bool GetCenterScaleFromSample(const vector<CSampleIDWeight> &sample, TDenseVector &center, TDenseMatrix &scale, double t, int posterior_type=POSTERIOR_HEATED); 

static TMatrix CreateProposalMatrix(int ndraws, CEquiEnergyModel &model, double lambda, const TElliptical *elliptical, int posterior_type)
{
	if (ndraws <= 0)
		return (TMatrix)NULL; 

	TMatrix proposal = CreateMatrix(ndraws, 2); 
	int dim = elliptical ? elliptical->dim : 0;
	double *draw = dim>0 ? new double[dim] : NULL; 

	CSampleIDWeight sample; 
	sample.data.Resize(dim); 
	double log_density, log_posterior;
	for (int i=0; i<ndraws; i++)
	{
		log_density = draw ?  LogDensityElliptical_Radius(DrawElliptical(draw, (TElliptical *)elliptical), (TElliptical *)elliptical) : 0.0;
		ElementM(proposal, i, 0) = log_density; 
	
		for (int j=0; j<dim; j++)
			sample.data[j] = draw[j]; 
		sample.DataChanged(); 
		model.log_posterior_function(sample); 

		if (posterior_type == POSTERIOR_HEATED)
			log_posterior = sample.weight*lambda; 
		else 
			log_posterior = sample.reserved*lambda + (sample.weight-sample.reserved); 
		ElementM(proposal, i, 1) = log_posterior; 
	}

	if (draw)
		delete [] draw; 
	return proposal; 
}

static TMatrix CreatePosteriorMatrix(const vector<CSampleIDWeight> &sample, double lambda, const TElliptical *elliptical, int posterior_type)
{
	TMatrix posterior=CreateMatrix(sample.size(),2); 
	int dim = elliptical ? elliptical->dim:0; 

	double log_density; 
	for (int i=0; i<(int)sample.size(); i++)
	{
		log_density = dim> 0 ? LogDensityElliptical_Draw(sample[i].data.vector, (TElliptical *)elliptical) : 0.0; 
		ElementM(posterior, i, 0) = log_density; 
		if (posterior_type == POSTERIOR_HEATED)
			ElementM(posterior, i, 1) = sample[i].weight * lambda;
		else
			ElementM(posterior, i, 1) = sample[i].reserved * lambda + (sample[i].weight-sample[i].reserved); 
	}
	return posterior; 
}


static TDenseVector GetRadiusFromSample(const vector<CSampleIDWeight> &sample, const TDenseVector &center, const TDenseMatrix &scale)
{
	TDenseVector Radii(sample.size(), 0.0); 
	TDenseMatrix quadratic_form = MultiplyTranspose(scale, scale); 
	quadratic_form.Inverse(SOLVE_SVD);
	quadratic_form = (quadratic_form + Transpose(quadratic_form)) * 0.5; // force symmetry
	
	TDenseVector theta; 
	for (int i=0; i<(int)sample.size(); i++)
	{
		theta = sample[i].data - center; 
		Radii[i] =  sqrt(InnerProduct(theta, theta, quadratic_form)); 
	}
	return Radii; 
}

static bool GetCenterScaleFromSample(const vector<CSampleIDWeight> &sample, TDenseVector &center, TDenseMatrix &scale, double lambda, int posterior_type)
{
	if (sample.empty())
	{
		cerr << "GetCenterScaleFromSample() : sample empty." << endl; 
		return false; 
	}


	double max; 
	CSampleIDWeight mode; 
	TDenseVector sample_sum(sample[0].data.dim, 0.0); 
	TDenseMatrix sample_square(sample[0].data.dim, sample[0].data.dim, 0.0); 
	for (int i=0; i<(int)sample.size(); i++)
	{
		if (i==0 || (posterior_type == POSTERIOR_HEATED && sample[i].weight*lambda > max) || (posterior_type != POSTERIOR_HEATED && sample[i].reserved*lambda+(sample[i].weight-sample[i].reserved) > max) )
		{
			if (posterior_type == POSTERIOR_HEATED)
				max = sample[i].weight*lambda; 
			else
				max = sample[i].reserved*lambda + (sample[i].weight-sample[i].reserved); 
			mode = sample[i];  
		}
		sample_sum = sample_sum + sample[i].data; 
		sample_square = sample_square + OuterProduct(sample[i].data, sample[i].data); 
	}
	
	sample_sum = sample_sum * (1.0/(double)sample.size()); 
	sample_square = sample_square * (1.0/(double)sample.size()); 	

	// using mode
	center = mode.data; 
	sample_square = sample_square + OuterProduct(mode.data, mode.data); 
	sample_square = sample_square - OuterProduct(sample_sum, mode.data) - OuterProduct(mode.data, sample_sum);
	// using center
	// sample_square = sample_square - OuterProduct(sample_sum, sample_sum); 	
 
	sample_square = (sample_square+Transpose(sample_square))*0.5; 
	
	/*scale = Cholesky(sample_square); 
	scale = Transpose(scale); */
	
	// Eig analysis
	TDenseVector eValue(sample_square.rows,0.0); 
	TDenseMatrix eVector(sample_square.rows, sample_square.cols, 0.0); 
	Eig(eValue, eVector, sample_square); 
	for (int i=0; i<eValue.dim; i++)
		eValue[i] = sqrt(eValue[i]); 
	scale = MultiplyTranspose(eVector * DiagonalMatrix(eValue), eVector); 
	return true; ; 
}


double LogMDD(const vector<CSampleIDWeight> &posterior, CEquiEnergyModel &model, double t, int proposal_type, int posterior_type)
{	
	TDenseVector center(posterior[0].data.dim, 0.0);
        TDenseMatrix scale(posterior[0].data.dim, posterior[0].data.dim, 0.0);
        if(!GetCenterScaleFromSample(posterior, center, scale, t, posterior_type))
        {
                cerr << "Error occurred in GetCenterScaleFromSample()\n";
                abort();
        } 
	 
	TDenseVector R = GetRadiusFromSample(posterior, center, scale); 

	TVector center_vector = CreateVector(center.dim); 
	TMatrix scale_matrix = CreateMatrix(scale.rows, scale.cols); 
	TVector R_vector = CreateVector(R.dim); 
	for (int i=0; i<center.dim; i++)
		ElementV(center_vector, i) = center[i]; 
	for (int i=0; i<scale.rows; i++)
		for (int j=0; j<scale.cols; j++)
			ElementM(scale_matrix, i, j) = scale(i,j); 
	for (int i=0; i<R.dim; i++)
		ElementV(R_vector, i) = R[i]; 
	
	double low=0.1, high=0.9; 
	TElliptical *elliptical=(TElliptical*)NULL; 
	switch (proposal_type)
       	{
               	case USE_GAUSSIAN:
                       	elliptical=CreateEllipticalFromPosterior_Gaussian(R_vector,center.dim,center_vector,scale_matrix);
                       	break;
               	case USE_POWER:
                       	elliptical=CreateEllipticalFromPosterior_Power(R_vector,center.dim,center_vector,scale_matrix, high);
                       	break;
               	case USE_TRUNCATED_POWER:
        		elliptical=CreateEllipticalFromPosterior_TruncatedPower(R_vector,center.dim,center_vector,scale_matrix,low,high);
                       	break;
               	case USE_STEP:
                       	elliptical=CreateEllipticalFromPosterior_Step(R_vector,center.dim,center_vector,scale_matrix,low,high);
                       	break;
               	case USE_TRUNCATED_GAUSSIAN:
                       	elliptical=CreateEllipticalFromPosterior_TruncatedGaussian(R_vector,center.dim,center_vector,scale_matrix,low,high);
               		break;
               	case USE_UNIFORM:
                      	elliptical=CreateEllipticalFromPosterior_Uniform(R_vector,center.dim,center_vector,scale_matrix,low,high);
                       	break;
               	default:
                       	cerr << "proposal type can only be 1 (Gaussian), 2(power), 3 (truncated power), 4 (step), 5 (truncated gaussian), 6 (uniform).\n";
                       	abort();
                       	break;
        }

	TMatrix proposal_matrix=CreateProposalMatrix((int)posterior.size(), model, t, elliptical, posterior_type); 
	TMatrix posterior_matrix=CreatePosteriorMatrix(posterior, t, elliptical, posterior_type); 

	if (!proposal_matrix || !posterior_matrix)
	{
		cerr << "There are no proposal or posterior draws.\n"; 
		abort(); 
	}

	int in_P1, in_P2; 
	double mdd_mueller = ComputeLogMarginalDensity_Mueller(proposal_matrix, posterior_matrix, &in_P1, &in_P2); 
	double mdd_bridge = ComputeLogMarginalDensity_Bridge(proposal_matrix, posterior_matrix); 

	FreeVector(center_vector); 
	FreeMatrix(scale_matrix); 
	FreeVector(R_vector); 
	FreeMatrix(proposal_matrix); 
	FreeMatrix(posterior_matrix); 
	FreeElliptical(elliptical);

	return mdd_bridge; 
}

double CheckConvergency(std::vector<CSampleIDWeight> &samples, CEquiEnergyModel &model, int stage, int previous_stage,  double convergency_previous, double &average_consistency, double &std_consistency, double &LB_ESS, int posterior_type, int nGroup_NSE)
{
	if (nGroup_NSE > 1)
	{
		for (int i=0; i<(int)samples.size(); i++)
			samples[i].id = samples[i].id % nGroup_NSE; 
	}
	sort(samples.begin(), samples.end(), compare_CSampleIDWeight_BasedOnID);

	double lambda_previous = model.parameter->lambda[previous_stage];
        double lambda_current = model.parameter->lambda[stage];
	vector<double> weight(samples.size(), 0.0); 
	for(int i=0; i<(int)samples.size(); i++)
	{
		if (posterior_type == POSTERIOR_HEATED)
                        weight[i] = samples[i].weight*lambda_current;
                else 
                        weight[i] = samples[i].reserved*lambda_current + (samples[i].weight-samples[i].reserved);

                if (posterior_type == POSTERIOR_HEATED)
                        weight[i] -= samples[i].weight*lambda_previous - convergency_previous;
                else
                        weight[i] -= samples[i].reserved*lambda_previous + (samples[i].weight-samples[i].reserved) - convergency_previous;
	}

	vector<double>group_consistency; 
	double consistency, sum_weight; 
	int counter =0; 
	for (int i=0; i<(int)(samples.size()); i++)
	{
		if (i==0)
		{ 
			group_consistency.push_back(weight[i]); 
			counter = 1; 
			consistency = weight[i]; 
		}
		else if (samples[i].id > samples[i-1].id)
		{
			group_consistency[group_consistency.size()-1] = group_consistency[group_consistency.size()-1] - log((double)counter); 
			group_consistency.push_back(weight[i]); 
			counter = 1; 
			consistency = AddLogs(consistency, weight[i]); 
		}
		else 
		{
			group_consistency[group_consistency.size()-1] = AddLogs(group_consistency[group_consistency.size()-1], weight[i]);  
			counter ++; 
			consistency = AddLogs(consistency, weight[i]);
		}
	}
	
	LB_ESS = 0.0; 
	for (int i=0; i<(int)samples.size(); i++)
		LB_ESS += exp(2.0*(weight[i]-consistency)); 
	LB_ESS = 1.0/LB_ESS;  

	group_consistency[group_consistency.size()-1] = group_consistency[group_consistency.size()-1] - log((double)counter);
	consistency = consistency - log((double)(samples.size())); 
	
	average_consistency=0.0; 
	std_consistency=0.0; 
	for (int i=0; i<(int)(group_consistency.size()); i++)
	{
		average_consistency += group_consistency[i]; 
		std_consistency += group_consistency[i] * group_consistency[i]; 
	}	
	average_consistency = average_consistency/(double)(group_consistency.size()); 
	std_consistency = sqrt(std_consistency/(double)(group_consistency.size())-average_consistency*average_consistency); 
	
	return consistency; 
}

double LogMDD_Importance(const vector<CSampleIDWeight> &proposal, CEquiEnergyModel &model, int previous_stage, int current_stage, double logMDD_previous, int posterior_type)
{
	double lambda_previous = model.parameter->lambda[previous_stage]; 
	double lambda_current = model.parameter->lambda[current_stage]; 

	vector<double> weight(proposal.size(),0); 
	for (int i=0; i<(int)proposal.size(); i++)
	{
		if (posterior_type == POSTERIOR_HEATED)
			weight[i] = proposal[i].weight * lambda_current; 
		else
			weight[i] = proposal[i].reserved * lambda_current + (proposal[i].weight-proposal[i].reserved); 

		if (posterior_type == POSTERIOR_HEATED)
			weight[i] -= proposal[i].weight * lambda_previous - logMDD_previous; 
		else
			weight[i] -= proposal[i].reserved * lambda_previous + (proposal[i].weight-proposal[i].reserved) - logMDD_previous;  
	}
	double sum_weight = weight[0];
	for (int i=1; i<(int)weight.size(); i++)
		sum_weight = AddLogs(sum_weight, weight[i]); 
	// changed by DW - must "divide" by number of observations
	return sum_weight- log(weight.size()); 
} 
