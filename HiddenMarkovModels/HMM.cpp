#include "stdafx.h"
#include "HMM.h"

using namespace std;

HMM::HMM(void)
{
}


HMM::~HMM(void)
{
}

HMM::HMM(const std::vector<std::vector<int>> &corpus, const int &V, const int &K, const int &seed)
	: x(corpus), V(V), K(K), M(corpus.size())
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
	n_kv = vector<vector<int>>(K, vector<int>(V));
	n_ko = vector<int>(K);
	n_ok = vector<int>(K);
	for(int j=0; j<M; ++j){
		int N_j = N[j];
		for(int i=0; i<N_j-1; ++i){
			int k = z[j][i];
			int k_next = z[j][i+1];
			n_kl[k][k_next]++;
			n_ko[k]++;
			n_ok[k_next]++;
			int v = x[j][i];
			n_kv[k][v]++;
		}
		n_kv[z[j][N_j-1]][x[j][N_j-1]]++;
	}
}

void HMM::train(int iter)
{
	sample_z();
}

void HMM::sample_z(void)
{
}

void HMM::show_parameters(void)
{
}

double HMM::calc_perplexity(void)
{
	return 0.0;
}