#include <vector>
#include "dw_dense_matrix.hpp"
#include "sbvar.hpp"

using namespace std; 

// Note:: SBVAR does not support restrictions
vector<TIndex> SBVAR::ConstructBlocks(int block_scheme) const
{
	vector<TIndex> blocks; 
	switch(block_scheme)
	{
		case 0: 
		{	// no blocking
			if (NumberParameters())
				blocks.push_back( TIndex(0, NumberParameters()-1) ); 
			break; 
		}
		case 1: 
		{	// one block for A0
			// another block for Aplus
			if (NumberVariables() )
				blocks.push_back( TIndex(0, NumberVariables()*NumberVariables()-1) );
			if (NumberVariables() && NumberPredetermined()) 
				blocks.push_back( TIndex(NumberVariables()*NumberVariables(), NumberVariables()*NumberPredetermined()-1) ); 
			break; 
		}
		case 2: 
		{	// one block for A0
			// one block for each row of Aplus
			if (NumberVariables())
				blocks.push_back(TIndex(0, NumberVariables()*NumberVariables()-1) ); 
			if (NumberPredetermined())
			{
				for (int ii=0; ii<NumberVariables(); ii++)
					blocks.push_back( TIndex(NumberVariables()*NumberVariables()+ii*NumberPredetermined(), NumberVariables()*NumberVariables()+(ii+1)*NumberPredetermined()-1) ); 
			}
			break; 
		}
		default: 
			blocks.clear(); 
	}
	return blocks; 
}

// Note: SBVAR_symmetric_linear allows for restriction of the form
//            A0(i,:)' = U(i) * b(i)
//            Aplus(i,:)' = V(i) * g(i)
//       U and V must have orthonormal columns
vector<TIndex> SBVAR_symmetric_linear::ConstructBlocks(int block_scheme) const
{
	vector<TIndex> blocks; 
	switch (block_scheme)
	{
		case 0: 
		{	// no blocksing
			if( NumberParameters())
				blocks.push_back( TIndex(0, NumberParameters()-1) ); 
			break; 
		}
		case 1: 
		{	// one block for A0
			// another block for Aplus
			for (int ii=0; ii<NumberVariables(); ii++)
			{
				if ( dim_b[ii] && blocks.empty() )
					blocks.push_back( TIndex(begin_b[ii], begin_b[ii]+dim_b[ii]-1) ); 
				else if (dim_b[ii])
					blocks[0] += TIndex(begin_b[ii], begin_b[ii]+dim_b[ii]-1); 
			}
			for (int ii=0; ii<NumberVariables(); ii++)
			{
				if (dim_g[ii] && blocks.size() < 2)
					blocks.push_back( TIndex(begin_g[ii], begin_g[ii]+dim_g[ii]-1) ); 
				else if (dim_g[ii])
					blocks[1] += TIndex(begin_g[ii], begin_g[ii]+dim_g[ii]-1); 
			}
			break; 
		}
		case 2: 
		{	// one block for A0
			// one block for each equation of the predetermined coefficients
			for (int ii=0; ii<NumberVariables(); ii++)
			{
				if (dim_b[ii] && blocks.empty() )
					blocks.push_back( TIndex(begin_b[ii], begin_b[ii]+dim_b[ii]-1) );
				else if (dim_b[ii])
					blocks[0] += TIndex(begin_b[ii], begin_b[ii]+dim_b[ii]-1); 
			}
			for (int ii=0; ii<NumberVariables(); ii++)
			{
				if (dim_g[ii])
					blocks.push_back( TIndex(begin_g[ii], begin_g[ii]+dim_g[ii]-1) );  
			}
			break; 
		}
		default:
			blocks.clear(); 
	}
	return blocks; 
}
