#include <vector>
#include <numeric>      // std::iota
#include <algorithm> 
#include <limits>
#include <iostream>
#include <cmath>
using std::vector;
using std::pair;
extern "C" {
	vector<size_t> sort_indexes(const vector<double> &v) {

	  vector<size_t> idx(v.size());
	  std::iota(idx.begin(), idx.end(), 0);

	  std::sort(idx.begin(), idx.end(),
		   [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

	  return idx;
	}
	/* ans = [is_leaf
			  val or feature
			  0 or thresh
	*/
	void fit_node(vector<vector<double>> &X, vector<double> &y, int min_samples_split, int min_samples_leaf, int max_depth, int depth, int node_id, float *ans, int ans_size) {
		int size = y.size(), double_size = sizeof(double), n_features = X.size();
		double inf = std::numeric_limits<double>::infinity();
		int n_nodes = (1 << (max_depth + 1)) - 1;
		vector<vector<int>> indexes(n_nodes, vector<int>());
		indexes[0].resize(y.size());
		std::iota(indexes[0].begin(), indexes[0].end(), 0);
		int offset = 0;
		vector<vector<double>> uniques(n_features, vector<double>());
		for (int i = 0; i < n_features; ++i) {
			auto sorted = sort_indexes(X[i]);
			for (int j = 0; j < sorted.size() - 1; ++j) {
				if (X[i][sorted[j]] != X[i][sorted[j + 1]]) {
					uniques[i].push_back(X[i][sorted[j]]);
				}
			}
			uniques[i].push_back(sorted[sorted.size() - 1]);
		}
		for (int j = 0; j <= max_depth; ++j) {
			int cur_nodes = (1 << j);
			if (j == max_depth) {
				for (int i = 0; i < cur_nodes; ++i) {
					ans[offset + i] = 1;
					double val = 0.0;
					vector<double> cur_y(indexes[offset + i].size());
					for (int m = 0; m < indexes[offset + i].size(); ++m) {
						cur_y[m] = y[indexes[offset + i][m]];
					}
					val =  std::accumulate(cur_y.begin(), cur_y.end(), 0.0);
					ans[offset + i + ans_size] = (val / cur_y.size()) * sqrt( ((double)cur_y.size()) / (cur_y.size() + 500)); //regularization, k = 500
				}	
				return;
			}
			vector<vector<double>> loss_thresholds(n_features, vector<double>());
			for (int i = 0; i < n_features; ++i) {
				loss_thresholds[i].resize(uniques[i].size());
			}

            for (int k = 0; k < cur_nodes; ++k) {
				vector<vector<double>> cur_X(n_features, vector<double>(indexes[k + offset].size()));
				vector<double> cur_y(indexes[k + offset].size());
				
				for (int m = 0; m < indexes[offset + k].size(); ++m) {
					for (int i = 0; i  < n_features; ++i) {
						cur_X[i][m] = X[i][indexes[offset + k][m]];
					}
					cur_y[m] = y[indexes[offset + k][m]];
				}					;

				for (int i = 0; i < n_features; ++i) {		
					vector<size_t> coefs = sort_indexes(cur_X[i]);
					vector<double> &x = cur_X[i];
					vector<double> cur_x, cur_y;
					int size = indexes[k + offset].size();
					for (int j = 0; j < size; ++j) {
						cur_x.push_back(x[coefs[j]]);
						cur_y.push_back(y[coefs[j]]);
					}
			
					vector<double> left_y_sum, left_y_sq_sum, right_y_sum, right_y_sq_sum, x_unique;
					
					double sum = 0, sq_sum = 0;
					vector<int> sizes_left, sizes_right;
					for (int j = 0; j < size - 1; ++j) {
						sum += cur_y[j];
						sq_sum += cur_y[j] * cur_y[j];
						if (cur_x[j] != cur_x[j + 1]) {
							left_y_sum.push_back(sum);
							left_y_sq_sum.push_back(sq_sum);
							sizes_left.push_back(j + 1);
							x_unique.push_back(cur_x[j]);
						}
					}
				
					sum += cur_y[size - 1];
					sq_sum += cur_y[size - 1] *  cur_y[size - 1];
					left_y_sum.push_back(sum);
					left_y_sq_sum.push_back(sq_sum);
					sizes_left.push_back(size);
					x_unique.push_back(cur_x[size -1]);
					int last_y_el = left_y_sum.size() - 1;
					for (int j = 0; j <= last_y_el; ++j) {
						sizes_right.push_back(sizes_left[last_y_el] - sizes_left[j]);
						right_y_sum.push_back(left_y_sum[last_y_el] - left_y_sum[j]);
						right_y_sq_sum.push_back(left_y_sq_sum[last_y_el] - left_y_sq_sum[j]);
					}
					sizes_right[last_y_el] = 1;
					vector<double> mse;
					for (int j = 0; j <= last_y_el; ++j) {
						double cur_mse = left_y_sq_sum[j] - left_y_sum[j] * left_y_sum[j] / sizes_left[j];
						cur_mse += right_y_sq_sum[j] - right_y_sum[j] * right_y_sum[j] / sizes_right[j];
						mse.push_back(cur_mse);
					
					}	
					int u = 0, j = 0;
					while(j < uniques[i].size() && uniques[i][j] <= x_unique[0]) {
						loss_thresholds[i][j] += mse[mse.size() - 1];
						j++;
					}
					u++;
                    for (; j < uniques[i].size() && u < x_unique.size();) {
						while(j < uniques[i].size() && uniques[i][j] <= x_unique[u]) {
							loss_thresholds[i][j] += mse[u-1];						

							j++;
						}
						u++;
					}
					for (; j < uniques[i].size(); ++j){
						loss_thresholds[i][j] += mse[mse.size() - 1];
					}
					
				}
			}
			vector<double> best_mses(n_features);
			int best_feature = 0;
			for (int i = 0; i < n_features; ++i) {
			
				best_mses[i] = *std::min_element(loss_thresholds[i].begin(), loss_thresholds[i].end());
		
			}
			for (int i = 0; i < n_features; ++i) {
				if (uniques[i].size() == 1) {
					best_mses[i] = inf;
				}
			}
			best_feature = std::min_element(best_mses.begin(), best_mses.end()) - best_mses.begin();
			int ind = std::min_element(loss_thresholds[best_feature].begin(), loss_thresholds[best_feature].end()) -
			loss_thresholds[best_feature].begin();
			double thresh = uniques[best_feature][ind];
			for (int i = 0; i < cur_nodes; ++i) {
				ans[offset + i] = 0;
				ans[offset + i + ans_size] = best_feature;
				ans[offset + i + 2 * ans_size] = thresh;
				for (int m = 0; m < indexes[offset + i].size(); ++m) {
					if (X[best_feature][indexes[offset + i][m]] >= thresh) {
						indexes[2 * (offset + i) + 1].push_back(indexes[offset + i][m]);
					} else {
						indexes[2 * (offset + i) + 2].push_back(indexes[offset + i][m]);
					}
				}
				if (indexes[2 * (offset + i) + 1].size() == 0) {
					for (int m = 0; m < indexes[2 * (offset + i) + 2].size(); ++m) {
						indexes[2 * (offset + i) + 1].push_back(indexes[2 * (offset + i) + 2][m]);
					}
				}
				if (indexes[2 * (offset + i) + 2].size() == 0) {
					for (int m = 0; m < indexes[2 * (offset + i) + 1].size(); ++m) {
						indexes[2 * (offset + i) + 2].push_back(indexes[2 * (offset + i) + 1][m]);
					}
				}

			}
			offset += cur_nodes;
		} 
	}
	
	
	void fit(double *X, double *y, int n_features, int n_objects,  int min_samples_split, int min_samples_leaf, int depth, float *ans, int ans_size) {
		vector<vector<double>>  data(n_features, vector<double>());
		vector<double> true_ans;
		for (int i = 0; i < n_features; ++i) {
			for (int j = 0; j < n_objects; ++j) {
				data[i].push_back(X[n_objects * i + j ]);
			}
		}
		for (int j = 0; j < n_objects; ++j) {
			true_ans.push_back(y[j]);
		}
		fit_node(data, true_ans,  min_samples_split, min_samples_leaf, depth, 0, 0, ans, ans_size);
	}
}
		
