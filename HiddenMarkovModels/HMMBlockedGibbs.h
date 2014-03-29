#pragma once
class HMMBlockedGibbs
{
public:
	HMMBlockedGibbs(void);
	~HMMBlockedGibbs(void);
	HMMBlockedGibbs(const std::vector<std::vector<int>> &corpus,
								const int &V,
								const int &K,
								const double &ALPHA,
								const double &BETA,
								const int &seed = 0);
	
	void train(int iter);
	void sample_pi(void);
	void sample_A(void);
	void sample_B(void);
	void sample_Z(void);
	void calc_counts_of_z(void);
	void show_parameters(void);
	double calc_perplexity(void);
	std::vector<double> estimate_map_pi(void);
	std::vector<std::vector<double>> estimate_map_A(void);
	std::vector<std::vector<double>> estimate_map_B(void);
	void save_model_parameter(const std::string &filename);
	
	// dataset
	int M;				// number of sequences in the dataset 
	std::vector<int> N;	// N[j]: number of tokens in j-th sequence
	int N_total;		// total number of tokens in the dataset
	int V;				// size of vocabulary
	// model parameters
	std::vector<double> pi;				// multinomial distribution of initial state. pi[k] is the mass of k-th state
	std::vector<std::vector<double>> A;	// state transition matrix. A[k][l] is transition probability from state k to l
	std::vector<std::vector<double>> B;	// emission distribution. B[k][v] is emission probability of word v from k-th multinomial distribution B[k]
	// model hyperparameters
	int K;								// number of states
	double ALPHA;						// prior parameter for each A_k ~ Dir(alpha)
	double BETA;						// prior parameter for each B_k ~ Dir(beta)
	// random variables
	const std::vector<std::vector<int>> x;	// x[j][i] is a observation word of i-th token in j-th sequence
	std::vector<std::vector<int>> z;		// z[j][i] is a latent variable of i-th token in j-th sequence

	boost::mt19937 rgen;
};