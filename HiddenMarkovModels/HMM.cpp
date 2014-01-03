#include "stdafx.h"
#include "HMM.h"
#include "utility.hpp"

using namespace std;

HMM::HMM(void)
{
}


HMM::~HMM(void)
{
}

HMM::HMM(const std::vector<std::vector<int>> &corpus, const int &V, const int &K, const double &ALPHA, const double &BETA, const int &seed)
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

	// initialize count
	n_kl = vector<vector<int>>(K, vector<int>(K));
	n_ko = vector<int>(K);
	n_kv = vector<vector<int>>(K, vector<int>(V));
	n_k = vector<int>(K);
	for(int j=0; j<M; ++j){
		int N_j = N[j];
		for(int i=0; i<N_j-1; ++i){
			int k = z[j][i];
			int k_next = z[j][i+1];
			n_kl[k][k_next]++;
			n_ko[k]++;
			int v = x[j][i];
			n_kv[k][v]++;
			n_k[k]++;
		}
		n_kv[z[j][N_j-1]][x[j][N_j-1]]++;
		n_k[z[j][N_j-1]]++;
	}
}

void HMM::train(int iter)
{
	sample_z();
}

void HMM::sample_z(void)
{
	for(int j=0; j<M; ++j){
		int N_j = N[j];

		// resample the first token
		{
			int v = x[j][0];
			int k_current = z[j][0];
			int k_next = z[j][1];
			n_kl[k_current][k_next]--;
			n_ko[k_current]--;
			n_kv[k_current][v]--;
			n_k[k_current]--;

			// according to "A comparison of Bayesian extimators for unsupervised Hidden Markov Model POS taggers"
			// without second term in Fig. 1
			vector<double> p_k(K);
			for(int k=0; k<K; ++k){
				// p(x_ji = v | z_ji = k, -)
				double term_1 = (n_kv[k][v] + BETA) / (n_k[k] + V * BETA);
				// p(z_{j,i+1} | p_ji = k, -)
				double term_3  = (n_kl[k][k_next] + ALPHA) / (n_ko[k] + K * ALPHA);
				p_k[k] = term_1 * term_3;
			}
			int k_new = util::multinomialByUnnormalizedParameters(rgen, p_k);
			z[j][0] = k_new;
			n_kl[k_new][k_next]++;
			n_ko[k_new]++;
			n_kv[k_new][v]++;
			n_k[k_new]++;
		}

		// resample all tokens except first and last token
		for(int i=1; i<N_j-1; ++i){
			int v = x[j][i];
			int k_prev = z[j][i-1];
			int k_current = z[j][i];
			int k_next = z[j][i+1];
			n_kl[k_prev][k_current]--;
			n_ko[k_prev]--;
			n_kl[k_current][k_next]--;
			n_ko[k_current]--;
			n_kv[k_current][v]--;
			n_k[k_current]--;

			// according to Fig. 1 in "A comparison of Bayesian extimators for unsupervised Hidden Markov Model POS taggers"
			vector<double> p_k(K);
			for(int k=0; k<K; ++k){
				// p(x_ji = v | z_ji = k, -)
				double term_1 = (n_kv[k][v] + BETA) / (n_k[k] + V * BETA);
				// p(z_ji = k | p_{j,i-1}, -)
				double term_2 = (n_kl[k_prev][k] + ALPHA);
				// p(z_{j,i+1} | p_ji = k, -)
				double term_3_numerator  = n_kl[k][k_next] + ALPHA + ((k_prev == k && k == k_next) ? 1 : 0);
				double term_3_denomiator = n_ko[k] + K * ALPHA + ((k_prev == k) ? 1 : 0);
				double term_3 = term_3_numerator / term_3_denomiator;
				p_k[k] = term_1 * term_2 * term_3;
			}
			int k_new = util::multinomialByUnnormalizedParameters(rgen, p_k);
			z[j][i] = k_new;
			n_kl[k_prev][k_new]++;
			n_ko[k_prev]++;
			n_kl[k_new][k_next]++;
			n_ko[k_new]++;
			n_kv[k_new][v]++;
			n_k[k_new]++;
		}
		
		// resample the last token
		{
			int i = N_j - 1;
			int v = x[j][i];
			int k_prev = z[j][i-1];
			int k_current = z[j][i];
			n_kl[k_prev][k_current]--;
			n_ko[k_prev]--;
			n_kv[k_current][v]--;
			n_k[k_current]--;

			// according to "A comparison of Bayesian extimators for unsupervised Hidden Markov Model POS taggers"
			// without 3rd term in Fig. 1
			vector<double> p_k(K);
			for(int k=0; k<K; ++k){
				// p(x_ji = v | z_ji = k, -)
				double term_1 = (n_kv[k][v] + BETA) / (n_k[k] + V * BETA);
				// p(z_ji = k | p_{j,i-1}, -)
				double term_2 = (n_kl[k_prev][k] + ALPHA);
				p_k[k] = term_1 * term_2;
			}
			int k_new = util::multinomialByUnnormalizedParameters(rgen, p_k);
			z[j][i] = k_new;
			n_kl[k_prev][k_new]++;
			n_ko[k_prev]++;
			n_kv[k_new][v]++;
			n_k[k_new]++;
		}
	}
}

void HMM::show_parameters(void)
{
}

double HMM::calc_perplexity(void)
{
	return 0.0;
}

vector<vector<double>> HMM::estimate_map_A(void)
{
	vector<vector<double>> A(K, vector<double>(K));
	for(int k=0; k<K; ++k){
		for(int l=0; l<K; ++l){
			A[k][l] = (n_kl[k][l] + ALPHA) / (n_ko[k] + K * ALPHA);
		}
	}

	return A;
}

vector<vector<double>> HMM::estimate_map_B(void)
{
	vector<vector<double>> B(K, vector<double>(V));
	for(int k=0; k<K; ++k){
		for(int v=0; v<V; ++v){
			B[k][v] = (n_kv[k][v] + BETA) / (n_k[k] + V * BETA);
		}
	}

	return B;
}