#include <vector>
#include "dw_dense_matrix.hpp"
#include "tvsbvar.hpp"
using namespace std; 

vector<TIndex > TVSBVAR:: ConstructBlocks(int block_scheme)
{
	vector<TIndex >blocks; 
	switch (block_scheme)
	{
		case 0:
		{	// no blocking 
			blocks.push_back( TIndex(0,NumberParameters()-1) ); 
			break;  
		}
		case 1:
		{	// separate block for each base regime, one block each for the
			// multiplicative parameters, the additive parameters, and 
			// the transition matrix parameters
			for (int kk=0; kk<BaseRestriction().n_regimes; kk++)
			{
				if ( BaseRestriction().dim[kk] )
					blocks.push_back( TIndex(BaseRestriction().offset[kk], BaseRestriction().offset[kk]+BaseRestriction().dim[kk]-1) ); 
			}
			if (MultiplicativeRestriction().n_free)
				blocks.push_back( TIndex(MultiplicativeRestriction().offset[0], MultiplicativeRestriction().offset[0]+MultiplicativeRestriction().n_free-1) ); 
			if (AdditiveRestriction().n_free)
				blocks.push_back( TIndex(AdditiveRestriction().offset[0], AdditiveRestriction().offset[0]+AdditiveRestriction().n_free-1) ); 
			if (NumberParameters_RegimeProcess())
				blocks.push_back( TIndex(Offset_RegimeProcessParameters(), Offset_RegimeProcessParameters()+NumberParameters_RegimeProcess()-1) ); 				

			break; 
		}
		case 2:
		{	// One block for all VAR parameters and one for transition
			// parameters
			if (NumberParameters_VAR())
				blocks.push_back(TIndex(0, NumberParameters_VAR()-1)); 
			if (NumberParameters_RegimeProcess() )
				blocks.push_back(TIndex(Offset_RegimeProcessParameters(), Offset_RegimeProcessParameters()+NumberParameters_RegimeProcess()-1)); 
			break; 
		}
		case 3:
		{	// two blocks for each base regime, one fore the 
			// contemporaneous coefficients and one for the pre-
			// determined coefficients
			// one block for multiplicative parameters
			// one block for additive parameters
			// one block for transition parameters
			for (int kk=0; kk<BaseRestriction().n_regimes; kk++)
			{
				for (int ii=0; ii< NumberVariables(); ii++)
				{
					if (BaseRestriction().dim_a[kk][ii] || BaseRestriction().dim_xi[kk][ii]) 
					{
						blocks.push_back( TIndex(BaseRestriction().offset_a[kk][0], BaseRestriction().offset_a[kk][NumberVariables()-1]+BaseRestriction().dim_a[kk][NumberVariables()-1]-1) ); 
						blocks[blocks.size()-1] += TIndex(BaseRestriction().offset_xi[kk][0], BaseRestriction().offset_xi[kk][NumberVariables()-1]+BaseRestriction().dim_xi[kk][NumberVariables()-1]-1);
						break; 
					}
				}
				for (int ii=0; ii<NumberVariables(); ii++)
				{
					if (BaseRestriction().dim_aplus[kk][ii])
					{
						blocks.push_back( TIndex(BaseRestriction().offset_aplus[kk][0], BaseRestriction().offset_aplus[kk][NumberVariables()-1]+BaseRestriction().dim_aplus[kk][NumberVariables()-1]-1) );
						break; 
					}
				}
			} 
			if (MultiplicativeRestriction().n_free)
				blocks.push_back( TIndex(MultiplicativeRestriction().offset[0], MultiplicativeRestriction().offset[0]+MultiplicativeRestriction().n_free-1) );  
			if (AdditiveRestriction().n_free)
				blocks.push_back( TIndex(AdditiveRestriction().offset[0], AdditiveRestriction().offset[0]+AdditiveRestriction().n_free-1) ); 
			if (NumberParameters_RegimeProcess() )
				blocks.push_back( TIndex(Offset_RegimeProcessParameters(), Offset_RegimeProcessParameters()+NumberParameters_RegimeProcess()-1) ); 
                        break;
		}
		case 4:
		{	// multiple blocks for each base regime
			// one for contemporaneous coefficients and one for each 
			// equation of the predetermined coefficients
			// one block for the multiplicative parameters
			// one block for the additive parameters
			// one block for transition parameters
			for (int kk=0; kk<BaseRestriction().n_regimes; kk++)
			{
				for (int ii=0; ii<NumberVariables(); ii++)
				{
					if (BaseRestriction().dim_a[kk][ii] || BaseRestriction().dim_xi[kk][ii])
					{
						blocks.push_back( TIndex(BaseRestriction().offset_a[kk][0],BaseRestriction().offset_a[kk][NumberVariables()-1]+BaseRestriction().dim_a[kk][NumberVariables()-1]-1) ); 
                                                blocks[blocks.size()-1] += TIndex(BaseRestriction().offset_xi[kk][0],BaseRestriction().offset_xi[kk][NumberVariables()-1]+BaseRestriction().dim_xi[kk][NumberVariables()-1]-1);

						break; 
					}
				}
				for (int ii=0; ii<NumberVariables(); ii++)
				{
					if (BaseRestriction().dim_aplus[kk][ii])
						blocks.push_back( TIndex(BaseRestriction().offset_aplus[kk][ii], BaseRestriction().offset_aplus[kk][ii]+BaseRestriction().dim_aplus[kk][ii]-1) );
				}
				if (MultiplicativeRestriction().n_free)
					blocks.push_back( TIndex(MultiplicativeRestriction().offset[0],MultiplicativeRestriction().offset[0]+MultiplicativeRestriction().n_free-1) );
				if (AdditiveRestriction().n_free)
					blocks.push_back( TIndex(AdditiveRestriction().offset[0],AdditiveRestriction().offset[0]+AdditiveRestriction().n_free-1) ); 
				if (NumberParameters_RegimeProcess())
					blocks.push_back( TIndex(Offset_RegimeProcessParameters(), Offset_RegimeProcessParameters()+NumberParameters_RegimeProcess()-1) ); 
			}
			break; 
		} 
		case 5:
		{	// multiple blocks for each base regime
			// one for contemporaneous coefficients and 
			// one for each equation of the predetermined coefficients
			// one block for the multiplicative parameters
			// one block for the additive parameters
			// one block for transition parameters
			// one block for hyper parameters
			for (int kk=0; kk<BaseRestriction().n_regimes; kk++)
			{
				for (int ii=0; ii<NumberVariables(); ii++)
				{
					if (BaseRestriction().dim_a[kk][ii] || BaseRestriction().dim_xi[kk][ii])
					{
						blocks.push_back( TIndex(BaseRestriction().offset_a[kk][0],BaseRestriction().offset_a[kk][NumberVariables()-1]+BaseRestriction().dim_a[kk][NumberVariables()-1]-1) );  
						blocks[blocks.size()-1] += TIndex(BaseRestriction().offset_xi[kk][0],BaseRestriction().offset_xi[kk][NumberVariables()-1]+BaseRestriction().dim_xi[kk][NumberVariables()-1]-1); 
						break; 
					}
				}
				for (int ii=0; ii<NumberVariables(); ii++)
				{
					if (BaseRestriction().dim_aplus[kk][ii])
						blocks.push_back(TIndex(BaseRestriction().offset_aplus[kk][ii], BaseRestriction().offset_aplus[kk][ii]+BaseRestriction().dim_aplus[kk][ii]-1) ); 
				}
				if (MultiplicativeRestriction().n_free)
					blocks.push_back( TIndex(MultiplicativeRestriction().offset[0],MultiplicativeRestriction().offset[0]+MultiplicativeRestriction().n_free-1) ); 
				if (AdditiveRestriction().n_free)
                        		blocks.push_back( TIndex(AdditiveRestriction().offset[0],AdditiveRestriction().offset[0]+AdditiveRestriction().n_free-1) );
				if (NumberParameters_RegimeProcess())
					blocks.push_back( TIndex(Offset_RegimeProcessParameters(), Offset_RegimeProcessParameters()+NumberParameters_RegimeProcess()-1) ); 
				if (NumberParameters_HyperParameter())
					blocks.push_back( TIndex(Offset_HyperParameters(), Offset_HyperParameters()+NumberParameters_HyperParameter()-1) ); 
			}
			break; 
		} 
		
		default:
			blocks.clear(); 
	}
	return blocks; 
}
