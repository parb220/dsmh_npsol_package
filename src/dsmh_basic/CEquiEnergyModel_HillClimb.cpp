#include <cmath>
#include "prcsn.h"
#include "dw_rand.h"
#include "CSampleIDWeight.hpp"
#include "CEESParameter.hpp"
#include "CStorageHead.hpp"
#include "CMetropolis.hpp"
#include "CEquiEnergyModel.hpp"
#include "dw_dense_matrix.hpp"

extern "C" {
	void npoptn_(char *, int);
	void npsol_(int *n, int *nclin, int *ncnln, int *ldA, int *ldJ, int *ldR, double *A, double *bl, double *bu, void *funcon(int *mode, int *ncnln, int *n, int *ldJ, int *needc, double *x, double *c, double *cJac, int *nstate), void *funobj(int *mode, int *n, double *x, double *f, double *g, int *nstate), int *inform, int *iter, int *istate, double *c, double *cJac, double *clamda, double *f, double *g, double *R, double *x, int *iw, int *leniw, double *w, int *lenw);
}

CEquiEnergyModel* MinusLogPosterior_NPSOL::model; 

void *MinusLogPosterior_NPSOL::function(int *mode, int *n, double *x, double *f, double *g, int *nstate)
{
	*f = -model->log_posterior_function(x,*n); 
}

std::vector<CSampleIDWeight> CEquiEnergyModel::HillClimb_NPSOL(int nSolution, int optimization_iteration, int perturbation_iteration, double perturbation_scale, double scale, const TDenseVector &start_point)
{
	const string COLD_START = string("Cold Start");
        const string NO_PRINT_OUT = string("Major print level = 0");
        const string DERIVATIVE_LEVEL = string("Derivative level = 0");
	const string HESSIAN = string("Hessian = Yes"); 

	// npsol 
	MinusLogPosterior_NPSOL::model = this; 
	int n = current_sample.data.dim; // sample dimension
	// linear constraints
	int nclin = 0; 	// number of linear constraints
	int ldA = nclin > 1 ? nclin : 1; 	// number of rows of A 
	double *A = new double[ldA*n]; 
	
	// nonlinear constraints	
	int ncnln = 0; 	// number of nonlinear constraints	
	int ldJ = ncnln > 1 ? ncnln : 1; 	// number of rows of cJac; 
	double *cJac = new double[ldJ*n]; 
	
	// R provides the upper-triangular Cholesky factor of the initial approximation
	// of the Hessian of the Lagrangian
	int ldR = n; 	// number of rows of R;
	double *R = new double[ldR*n];   
	
	int nctotal = n + nclin + ncnln; 
	int *istate = new int[nctotal]; // array indicating whether the constaits are effective
	double *bl = new double[nctotal]; // lower bounds for samples, A and cJac
	double *bu = new double[nctotal]; // upper bounds for samples, A and cJac
	for (int i=0; i<n; i++)
	{
		bl[i] = MINUS_INFINITY; 
		bu[i] = PLUS_INFINITY; 
	}
	double *clambda = new double [nctotal]; // lagragin parameters of the constraints	
	for (int i=0; i<nctotal; i++)
		clambda[i] = 0; 

	// work space
	int leniw = 3*n + nclin + 2*ncnln; 
	int *iw = new int [leniw]; 	// integer work space
	int lenw; 
	if (nclin == 0 && ncnln == 0)
		lenw = 20*n; 
	else if (ncnln == 0)
		lenw = 2*n*n + 20*n + 11*nclin; 
	else 
		lenw = 2*n*n + n*nclin + 2*n*ncnln + 20*n + 11*nclin + 21*ncnln;
	double *w = new double[lenw]; 	// floating work space

	// initial estimate of the solution 
	double *x = new double[n]; 	

	// returning value of npsol
	int inform, iter;
	double *c = new double[1]; // because ncnln = 0
	double f; 	// value of objective f(x) at the final iterate
	double *g = new double[n]; // objective gradient	   

	int nAccpt = 0; 

	gmm_mean.clear(); 
	gmm_covariance_sqrt.clear();
	gmm_covariance_sqrt_log_determinant.clear(); 
	gmm_covariance_sqrt_inverse.clear(); 
 
	double log_posterior_before_perturbation, log_posterior_after_perturbation, log_posterior_optimal; 
	TDenseVector peak(n,0.0), perturbation(n,0.0), x_plus_perturbation(n,0.0);  
	TDenseMatrix hessian_cholesky(n,n,0.0); 
	std::vector<CSampleIDWeight> solution; 

	try {
		for (int i=0; i<nSolution; i++)
		{
			if (start_point.dim)
				memcpy(x, start_point.vector, sizeof(double)*start_point.dim); 
			else 
			{
				log_posterior_before_perturbation = MINUS_INFINITY; 
				while (log_posterior_before_perturbation <= MINUS_INFINITY)
				{
					if(!DrawParametersFromPrior(x))
						throw dw_exception("CEquiEnergyModel::HillClimb_NPSOL : DrawSampleFromPrior() error occurred");
					log_posterior_before_perturbation = log_posterior_function(x, n);
				}
			}

			for (int ii=0; ii<optimization_iteration; ii++)
			{
				perturbation.RandomNormal(); 
				// Perturbation
				for (int jj=0; jj<perturbation_iteration; jj++)
				{
					memcpy(x_plus_perturbation.vector, x, sizeof(double)*n); 
					x_plus_perturbation += perturbation_scale * perturbation; 
					log_posterior_after_perturbation = log_posterior_function(x_plus_perturbation.vector, n);
					if (log_posterior_after_perturbation > log_posterior_before_perturbation)
					{
						memcpy(x, x_plus_perturbation.vector, sizeof(double)*n);
						log_posterior_before_perturbation = log_posterior_after_perturbation; 
					}	
					perturbation = perturbation*0.5; 	
				}
				
				// Optimization
				npoptn_((char*)DERIVATIVE_LEVEL.c_str(), DERIVATIVE_LEVEL.length());
                       	 	npoptn_((char*)NO_PRINT_OUT.c_str(), NO_PRINT_OUT.length());
                        	npoptn_((char*)COLD_START.c_str(), COLD_START.length());
                        	npoptn_((char*)HESSIAN.c_str(), HESSIAN.length()); 
                        	npsol_(&n, &nclin, &ncnln, &ldA, &ldJ, &ldR, A, bl, bu, NULL, MinusLogPosterior_NPSOL::function, &inform, &iter, istate, c, cJac, clambda, &f, g, R, x, iw, &leniw, w, &lenw);
				
				if (ii==0 || -f > log_posterior_optimal)
				{
					log_posterior_optimal = -f; 
					memcpy(peak.vector, x, sizeof(double)*n);
                        		memcpy(hessian_cholesky.matrix, R, sizeof(double)*n*n); 
                        		hessian_cholesky.column_major = true;
				}
			}

			// Accept the optimal solution
			if (log_posterior_optimal > MINUS_INFINITY)
			{
				nAccpt ++; 
				// Mean 
				TDenseVector copy(peak.dim); 
				copy.CopyContent(peak); 
				gmm_mean.push_back(copy);  
				// Hessian = inverse of covariance matrix
				TDenseMatrix hessian = Transpose(hessian_cholesky)*hessian_cholesky; 
				hessian = 0.5*(hessian+Transpose(hessian)); 
				TDenseMatrix eVectorHessian; 
				TDenseVector eValueHessian; 
				Eig(eValueHessian, eVectorHessian, hessian);  
				// hessian = eVectorHessian * DiagonalMatrix(eValueHessian) * eVectorHessian'
				// covariance_matrix = eVectorHessian * Inverse(DiagonalMatrix(eValueHessian)) *eVectorHessian';
			
				TDenseVector diffusedEValue(eValueHessian.dim,0.0), diffusedEValueInverse(eValueHessian.dim,0.0); 
				double covariance_sqrt_log_determinant = 0.0; 
				for (int i=0; i<eValueHessian.dim; i++)
				{
					diffusedEValue[i] = sqrt(scale/eValueHessian[i]); 
					diffusedEValueInverse[i] = sqrt(eValueHessian[i]/scale); 
					covariance_sqrt_log_determinant += -0.5*log(eValueHessian[i]/scale); 
				}

				gmm_covariance_sqrt.push_back(eVectorHessian*DiagonalMatrix(diffusedEValue)*Transpose(eVectorHessian)); 
				gmm_covariance_sqrt_inverse.push_back(eVectorHessian*DiagonalMatrix(diffusedEValueInverse)*Transpose(eVectorHessian)); 
				gmm_covariance_sqrt_log_determinant.push_back(covariance_sqrt_log_determinant); 

				solution.push_back(CSampleIDWeight(copy, 0, log_posterior_function(copy.vector, copy.dim), log_likelihood_function(copy.vector, copy.dim), true)); 
			}
		}
	}	
	catch( dw_exception& e )
	{
		cout << e.what() << endl; 
		delete [] g; 
		delete [] c;
		delete [] w; 
		delete [] iw; 
		delete [] istate;
		delete [] clambda; 
		delete [] bu; 
		delete [] bl;  
		delete [] R; 
		delete [] cJac; 
		delete [] A; 
		throw; 
	}

	delete [] g; 
	delete [] c;
	delete [] w; 
	delete [] iw; 
	delete [] istate;
	delete [] clambda; 
	delete [] bu; 
	delete [] bl;  
	delete [] R; 
	delete [] cJac; 
	delete [] A; 

	return solution; 
}

