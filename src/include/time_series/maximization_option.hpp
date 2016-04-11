#ifndef _MAXIMIZATION_OPTION_
#define _MAXIMIZATION_OPTION_

#include <vector>
#include "dw_dense_matrix.hpp"

class MaximizationOptions
{
public:
        std::vector<TIndex> blocks;
	int BlockScheme; 
        double PerturbationScale;
        int MaxPerturbationIterations;
        int MaxBlockIterations;
        int MaxOptimizationIterations;
	bool ConstantOptimization;
};

#endif
