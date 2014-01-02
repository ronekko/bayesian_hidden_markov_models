#pragma once
class HMM
{
public:
	HMM(void);
	~HMM(void);
	HMM(const std::vector<std::vector<int>> &corpus,
								const int &V,
								const int &K = 12,
								const int &seed = 0);
	
	void train(int iter);
	void show_parameters(void);
	double calc_perplexity(void);
	
	boost::mt19937 engine;
};

