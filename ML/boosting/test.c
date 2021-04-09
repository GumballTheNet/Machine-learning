#include <stdio.h>
#include <stdlib.h>
int simple_function(float *X_test, int len, int n_features, int *features, float *thresh, float *values, int *is_leaf, float *res, int *feature_ind) {
    int i = 0, j = 0;
    int node = 0, depth = 0;
    while (!is_leaf[node]) {
       depth += 1;
       node = node * 2 + 1;
    }
    node = 0;
    float mask[depth];
    int offset = (1 << (depth)) -1;
    depth = 0;
   
    while (!is_leaf[node]) {
       mask[depth] = thresh[node];

       depth += 1;
       node = node * 2 + 1;
    }

    for (i = 0; i < len; ++i) {
        int leaf_num = 0, node = 0;
        for (j = 0; j < depth; ++j) {
           leaf_num = leaf_num << 1;
           leaf_num = leaf_num | ((int) (X_test[i * n_features + feature_ind[features[node]]] < mask[j]));

           node = node * 2 + 1;
        }
	
	res[i] = values[offset + leaf_num];
    }
    return 0;
}

int boost_pred(float *X_test, int len, int n_features, int n_models, int tree_node_count,  int *features, float *thresh, float *values, int *is_leaf,
 float *coeffs, float *res, int *feature_ind, int feature_size) {
	int i = 0, j = 0;
    for (i = 0; i < n_models; ++i) {
		int current_tree = i * tree_node_count, cur_feature = i * feature_size  ;
		for (j = 0; j < len; ++j) {
				int node = 0;
				while (!is_leaf[current_tree + node]) {
					if (X_test[j * n_features + feature_ind[cur_feature + features[current_tree + node ]]] >  thresh[current_tree + node ])
						node = 2 * node + 1;
					else
						node = 2 * node + 2;
				}
				res[j] += coeffs[i] * values[current_tree + node];
		}
	}
	return 0;
}
