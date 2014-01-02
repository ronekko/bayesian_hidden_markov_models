#pragma once
class HMM
{
public:
	HMM(void);
	~HMM(void);
	HMM(const std::vector<std::vector<int>> &corpus,
								const int &V,
								const int &K = 10,
								const int &seed = 0);
	
	void train(int iter);
	void show_parameters(void);
	double calc_perplexity(void);
	
	// dataset
	int M;				// number of sequences in the dataset 
	std::vector<int> N;	// N[j]: number of tokens in j-th sequence
	int N_total;		// total number of tokens in the dataset
	// model parameters
	std::vector<double> pi;				// multinomial distribution of initial state. pi[k] is the mass of k-th state
	std::vector<std::vector<double>> A;	// state transition matrix. A[k][l] is transition probability from state k to l
	std::vector<std::vector<double>> B;	// emission distribution. B[k][v] is emission probability of word v from k-th multinomial distribution B[k]
	// model hyperparameters
	int K;			// number of states
	double alpha;	// prior parameter for each A_k ~ Dir(alpha)
	double beta;	// prior parameter for each B_k ~ Dir(beta)
	// random variables
	std::vector<std::vector<int>> x; // x[j][i] is a observation word of i-th token in j-th sequence
	std::vector<std::vector<int>> z; // z[j][i] is a latent variable of i-th token in j-th sequence
	// counts
	std::vector<std::vector<int>> n_kl; // frequencies of state transition from k to l in z

	boost::mt19937 engine;
};

