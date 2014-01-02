// HiddenMarkovModels : コンソール アプリケーションのエントリ ポイントを定義します。
//

#include "stdafx.h"
#include "HMM.h"
#include "utility.hpp"

//unsigned int fp_control_state = _controlfp(_EM_INEXACT, _MCW_EM);

using std::vector;
using std::string;

cv::Mat upsample(const cv::Mat &src, const int &scale)
{
	using namespace cv;
	int rows = src.rows * scale;
	int cols = src.cols * scale;
	Mat dst(rows, cols, src.type());

	for(int i=0; i<rows; ++i){
		for(int j=0; j<cols; ++j){
			dst.at<float>(i,j) = src.at<float>(i/scale,j/scale);
		}
	}

	return dst;
}

void showSticks(const vector<double> &stickLengths)
{
	using namespace cv;
	int h = 30;
	int w = 700;
	int K = stickLengths.size();
	Mat image = Mat::zeros(h, w, CV_8UC1);
	vector<int> breakingPoints(K);
	double sum = 0.0;
	for(int k=0; k<K; ++k){
		sum += stickLengths[k];
		breakingPoints[k] = sum * w;
	}
	int currentPoint = 0;
	for(int k=0; k<K; ++k){
		rectangle(image, Point(currentPoint, 0), Point(breakingPoints[k], h), Scalar((k%3)*128-1), CV_FILLED);
		currentPoint = breakingPoints[k];
	}
	imshow("Stick", image);
	waitKey(1);
}


// if 'stick' is specified, then the topics are sorted according to the stick lengths
void showTopics(const string &title, const vector<vector<double>> &phi, const int &numColsPerRow = 5, const vector<double> &stick = vector<double>())
{
	using namespace cv;
	bool use_stick = !!stick.size();
	const int K = phi.size();
	const int V = phi[0].size();
	const int COLS = numColsPerRow;
	const int ROWS = ceil(double(K) / double(COLS));
	vector<vector<double>> phi_tmp(phi.begin(), phi.end());
	vector<Mat> phiImages;
	Mat result(ROWS*60, COLS*60, CV_32FC1);

	if(use_stick){
		vector<std::pair<vector<double>, double>> phi_weighted;
		vector<double> stick_sorted(K);
		phi_weighted.reserve(K);
		for(int k=0; k<K; ++k){
			phi_weighted.push_back(make_pair(phi[k], stick[k]));
		}
		boost::sort(phi_weighted, [](const std::pair<vector<double>, double> &lhs, 
									 const std::pair<vector<double>, double> &rhs){
										 return lhs.second > rhs.second;
									});
		for(int k=0; k<K; ++k){
			phi_tmp[k] = phi_weighted[k].first;
			stick_sorted[k] = phi_weighted[k].second;
		}
		showSticks(stick_sorted);
	}
						
			
	
	for(int k=0; k<K; ++k){
		Mat phiImage(5, 5, CV_32FC1);
		for(int i=0; i<5; ++i){
			for(int j=0; j<5; ++j){
				phiImage.at<float>(i, j) = static_cast<float>(phi_tmp[k][i*5+j]);
			}
		}
		phiImages.push_back(upsample(phiImage, 10.0) * 5.0);
	}

	randu(result, Scalar(0.0), Scalar(1.0));

	for(int k=0; k<K; ++k){
		int row = k / COLS;
		int col = k % COLS;
		Mat roi = result(Rect(col*60+5, row*60+5, 50, 50));
		phiImages[k].copyTo(roi);
	}
	imshow(title, result);
	waitKey(1);
}


vector<vector<double>> createTopics(void)
{
	// 0.1 * 9 + 0.00625 * 16
	vector<vector<double>> topics;
	const int V = 25;

	for(int i=0; i<5; ++i){
		vector<double> phi(V, 0.00625);
		for(int j=0; j<5; ++j){
			phi[i*5+j] = 0.1;
			phi[j*5+i] = 0.1;
		}
		topics.push_back(phi);
	}

	for(int i=0; i<5; ++i){
		vector<double> phi(V, 0.00625);
		for(int j=0; j<5; ++j){
			phi[i*5+(4-j)] = 0.1;
			phi[j*5+(4-i)] = 0.1;
		}
		topics.push_back(phi);
	}
	
	{
		vector<double> phi(V, 0.00625);
		phi[0] = phi[4] = phi[6] = phi[8] = phi[12]
		= phi[16] = phi[18] = phi[20] = phi[24] = 0.1;
		topics[7] = phi;
	}

	return topics;
}


int _tmain(int argc, _TCHAR* argv[])
{
	using namespace std;
	const int M = 1000;
	const int N_mean = 400;
	const int V = 25;
	int K_max = 20;
	int K_init = 10;
	const double ALPHA = 1.0;
	boost::mt19937 engine;
	vector<vector<double>> theta(M);
	
	// create synthetic emission component distributions
	vector<vector<double>> topics = createTopics();
	showTopics("true topics", topics);
	cv::waitKey();

	const int K = topics.size();
	vector<boost::random::discrete_distribution<>> word_distributions(K);
	for(int k=0; k<K; ++k){
		word_distributions[k] = boost::random::discrete_distribution<>(topics[k]);
	}

	vector<vector<int>> corpus(M);
	vector<double> beta(K);
	//boost::iota(beta, 1);
	for(int k=0; k<K; ++k){
		beta[K-1-k] = k*2 + 1;
	}
	beta = util::l1_normalize(beta);

	for(auto bk:beta){
		cout << bk << " ";
	}cout << endl;
	vector<double> alpha(K, ALPHA / K);
	for(int k=0; k<K; ++k){
		alpha[k] = ALPHA * beta[k];
	}
	
	for(int m=0; m<M; ++m){
		int N_m = boost::poisson_distribution<>(N_mean)(engine);
		corpus[m].resize(N_m);

		vector<double> theta_m = util::dirichletRandom(engine, alpha);
		theta[m] = theta_m;

		boost::random::discrete_distribution<> discrete(theta_m);
		for(int i=0; i<N_m; ++i){
			int k = discrete(engine);
			int v = word_distributions[k](engine);
			corpus[m][i] = v;
		}
	}

	
	HMM model(corpus, V, K, 11);
	
	bool show_topics_with_sort = false;
	for(int i=0; i<10000; ++i){
		cout << "# " << i << " ##########" << endl;
		boost::timer timer;
		model.train(1);
		cout << "time: " << timer.elapsed() << endl;
		model.show_parameters();
		cout << "perplexity: " << model.calc_perplexity() << endl<<endl;
		
		//showTopics("topics", learner.phi,10);
	}
	return 0;
}

