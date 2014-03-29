#include "stdafx.h"
#include "utility.hpp"
#include "HMMBlockedGibbs.h"

using namespace std;

HMMBlockedGibbs::HMMBlockedGibbs(void)
{
}


HMMBlockedGibbs::~HMMBlockedGibbs(void)
{
}

HMMBlockedGibbs::HMMBlockedGibbs(const std::vector<std::vector<int>> &corpus, const int &V, const int &K, const double &ALPHA, const double &BETA, const int &seed)
	: x(corpus), V(V), K(K), ALPHA(ALPHA), BETA(BETA), M(corpus.size())
{
	rgen.seed(seed);

	N_total = 0;
	N = vector<int>(M);

	// initialize z
	z = vector<vector<int>>(M);
	for(int j=0; j<M; ++j){
		int N_j = x[j].size();
		N[j] = N_j;
		N_total += N_j;
		z[j] = vector<int>(N_j);
		for(int i=0; i<N_j; ++i){
			z[j][i] = boost::uniform_int<>(0, K-1)(rgen);
		}
	}
	// initialize pi, A, B
	pi = vector<double>(K);
	A = vector<vector<double>>(K, vector<double>(K));
	B = vector<vector<double>>(K, vector<double>(V));
}


void HMMBlockedGibbs::train(int iter)
{
	sample_pi();
	sample_A();
	sample_B();
	sample_Z();
}



void HMMBlockedGibbs::sample_pi(void)
{
	vector<int> n_0k = vector<int>(K); // counts for each state from first tokens of all sequences
	for(int j=0; j<M; ++j){
		n_0k[z[j][0]]++;
	}

	vector<double> dir_pi(K);
	for(int k=0; k<K; ++k){
		dir_pi[k] = n_0k[k] + ALPHA;
	}
	this->pi = util::dirichletRandom(rgen, dir_pi);
}

void HMMBlockedGibbs::sample_A(void)
{
	vector<vector<int>> n_kl = vector<vector<int>>(K, vector<int>(K)); // counts of state transition from k to l
	for(int j=0; j<M; ++j){
		int N_j = N[j];
		for(int i=0; i<N_j-1; ++i){
			int k = z[j][i];
			int k_next = z[j][i+1];
			n_kl[k][k_next]++;
		}
	}

	for(int k=0; k<K; ++k){
		vector<double> dir_A_k(K);
		for(int l=0; l<K; ++l){
			dir_A_k[l] = n_kl[k][l] + ALPHA;
		}
		this->A[k] = util::dirichletRandom(rgen, dir_A_k);
	}
}

void HMMBlockedGibbs::sample_B(void)
{
	vector<vector<int>> n_kv = vector<vector<int>>(K, vector<int>(V)); // counts of occurrence of word v from state k
	for(int j=0; j<M; ++j){
		int N_j = N[j];
		for(int i=0; i<N_j; ++i){
			int k = z[j][i];
			int v = x[j][i];
			n_kv[k][v]++;
		}
	}

	for(int k=0; k<K; ++k){
		vector<double> dir_B_k(V);
		for(int v=0; v<V; ++v){
			dir_B_k[v] = n_kv[k][v] + BETA;
		}
		this->B[k] = util::dirichletRandom(rgen, dir_B_k);
	}
}

void HMMBlockedGibbs::sample_Z(void)
{
	for(int j=0; j<M; ++j){
		int N_j = N[j];

		// forward calculation
		vector<vector<double>> F(N_j, vector<double>(K)); 
		{
			int v = x[j][0];
			double normalizer = 0.0;
			for(int k=0; k<K; ++k){
				F[0][k] = pi[k] * B[k][v];
				normalizer += F[0][k];
			}
			for(int k=0; k<K; ++k){
				F[0][k] /= normalizer;
			}
		}
		for(int i=1; i<N_j; ++i){
			int v = x[j][i];
			double normalizer = 0.0;
			for(int k=0; k<K; ++k){
				double FA_k = 0;
				for(int l=0; l<K; ++l){
					FA_k += A[l][k] * F[i-1][l];
				}
				F[i][k] = FA_k * B[k][v];
				normalizer += F[i][k];
			}
			for(int k=0; k<K; ++k){
				F[i][k] /= normalizer;
			}
		}

		// backward sampling
		int k_new = util::multinomialByUnnormalizedParameters(rgen, F.back());
		z[j].back() = k_new;
		for(int i=N_j-2; i>=0; --i){
			vector<double> p(K);
			int k_next = z[j][i+1];
			for(int k=0; k<K; ++k){
				p[k] = F[i][k] * A[k][k_next];
			}
			int k_new = util::multinomialByUnnormalizedParameters(rgen, p);
			z[j][i] = k_new;
		}
	}
}

void HMMBlockedGibbs::show_parameters(void)
{
}

double HMMBlockedGibbs::calc_perplexity(void)
{
	return 0.0;
}

vector<double> HMMBlockedGibbs::estimate_map_pi(void)
{
	vector<int> n_0k = vector<int>(K); // counts for each state from first tokens of all sequences
	for(int j=0; j<M; ++j){
		n_0k[z[j][0]]++;
	}

	vector<double> pi(K);
	for(int k=0; k<K; ++k){
		pi[k] = (n_0k[k] + ALPHA) / (M + K * ALPHA);
	}

	return pi;
}

vector<vector<double>> HMMBlockedGibbs::estimate_map_A(void)
{
	vector<vector<int>> n_kl = vector<vector<int>>(K, vector<int>(K)); // counts of state transition from k to l
	vector<int> n_ko = vector<int>(K); // counts of state transition from k to any
	for(int j=0; j<M; ++j){
		int N_j = N[j];
		for(int i=0; i<N_j-1; ++i){
			int k = z[j][i];
			int k_next = z[j][i+1];
			n_kl[k][k_next]++;
			n_ko[k]++;
		}
	}

	vector<vector<double>> A(K, vector<double>(K));
	for(int k=0; k<K; ++k){
		for(int l=0; l<K; ++l){
			A[k][l] = (n_kl[k][l] + ALPHA) / (n_ko[k] + K * ALPHA);
		}
	}

	return A;
}

vector<vector<double>> HMMBlockedGibbs::estimate_map_B(void)
{
	vector<vector<int>> n_kv = vector<vector<int>>(K, vector<int>(V)); // counts of occurrence of word v from state k
	vector<int> n_k = vector<int>(K); // counts of state k
	for(int j=0; j<M; ++j){
		int N_j = N[j];
		for(int i=0; i<N_j; ++i){
			int k = z[j][i];
			int v = x[j][i];
			n_kv[k][v]++;
			n_k[k]++;
		}
	}

	vector<vector<double>> B(K, vector<double>(V));
	for(int k=0; k<K; ++k){
		for(int v=0; v<V; ++v){
			B[k][v] = (n_kv[k][v] + BETA) / (n_k[k] + V * BETA);
		}
	}

	return B;
}

void HMMBlockedGibbs::save_model_parameter(const std::string &filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << "V" << V;
	fs << "K" << K;
	fs << "ALPHA" << ALPHA;
	fs << "BETA" << BETA;
	fs << "pi" << cv::Mat(estimate_map_pi());
	fs << "A" << util::vector_to_Mat(estimate_map_A());
	fs << "B" << util::vector_to_Mat(estimate_map_B());
}