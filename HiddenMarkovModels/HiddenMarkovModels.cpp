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
	const int W = ceil(sqrt(V));
	const int COLS = numColsPerRow;
	const int ROWS = ceil(double(K) / double(COLS));
	vector<vector<double>> phi_tmp(phi.begin(), phi.end());
	vector<Mat> phiImages;
	Mat result(ROWS*(W*10+10), COLS*(W*10+10), CV_32FC1);

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
		Mat phiImage(W, W, CV_32FC1);
		for(int i=0; i<W; ++i){
			for(int j=0; j<W; ++j){
				phiImage.at<float>(i, j) = static_cast<float>(phi_tmp[k][i*W+j]);
			}
		}
		phiImages.push_back(upsample(phiImage, 10.0) * 5.0);
	}

	randu(result, Scalar(0.0), Scalar(1.0));

	for(int k=0; k<K; ++k){
		int row = k / COLS;
		int col = k % COLS;
		Mat roi = result(Rect(col*(W * 10 + 10) + 5, row*(W * 10 + 10) + 5, W*10, W*10));
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


vector<vector<int>> create_corpus(void)
{
	using namespace std;
	const int M = 1000;
	const int N_mean = 400;
	const int V = 25;
	const int K = 10;
	const double ALPHA = 1.0 / K;
	const double BETA = 1.0 / V;
	boost::mt19937 rgen;
	
	// create synthetic emission component distributions
	vector<vector<double>> B = createTopics();
	// craete synthetic state transition matrix
	vector<vector<double>> A(K, vector<double>(K));
	std::array<double, 4> a = {0.85, 0.1, 0.025, 0.025};
	for(int k=0; k<K; ++k){
		for(int r=0; r<a.size(); ++r){
			int l = (k + r) % K;
			A[k][l] = a[r];
		}
	}
	{
		vector<vector<double>> A_(1, vector<double>(K*K));
		for(int k=0; k<K; ++k){
			for(int l=0; l<K; ++l){
				A_[0][k*K+l] = A[k][l];
			}
		}
		showTopics("true B", B, 5);
		showTopics("true A", A_, 5);
	}
	
	vector<boost::random::discrete_distribution<>> A_distributions(K);
	vector<boost::random::discrete_distribution<>> B_distributions(K);
	for(int k=0; k<K; ++k){
		A_distributions[k] = boost::random::discrete_distribution<>(A[k]);
		B_distributions[k] = boost::random::discrete_distribution<>(B[k]);
	}

	// generate a corpus according to the generative model of HMM
	vector<vector<int>> corpus(M);	
	for(int j=0; j<M; ++j){
		int N_j = boost::poisson_distribution<>(N_mean)(rgen);
		corpus[j].resize(N_j);
		
		// put uniform distribution for initial state
		int z_j0 = boost::uniform_int<>(0, K-1)(rgen);
		corpus[j][0] = B_distributions[z_j0](rgen);
		int z_prev = z_j0;
		for(int i=1; i<N_j; ++i){
			int k = A_distributions[z_prev](rgen);
			int v = B_distributions[k](rgen);
			corpus[j][i] = v;
			z_prev = k;
		}
	}

	return corpus;
}

int _tmain(int argc, _TCHAR* argv[])
{
	using namespace std;
	const int M = 1000;
	const int N_mean = 400;
	const int V = 25;
	const int K = 10;
	const double ALPHA = 1.0 / K;
	const double BETA = 1.0 / V;

	vector<vector<int>> corpus = create_corpus();

	HMM model(corpus, V, K, ALPHA, BETA,2);
	
	for(int i=0; i<10000; ++i){
		cout << "# " << i << " ##########" << endl;
		boost::timer timer;
		model.train(1);
		cout << "time: " << timer.elapsed() << endl;
		model.show_parameters();
		cout << "perplexity: " << model.calc_perplexity() << endl<<endl;
		
		{
			auto A = model.estimate_map_A();
			vector<vector<double>> A_(1, vector<double>(K*K));
			for(int k=0; k<K; ++k){
				for(int l=0; l<K; ++l){
					A_[0][k*K+l] = A[k][l];
				}
			}
			showTopics("estimated A", A_, 5);
			showTopics("estimated B", model.estimate_map_B(), 5);
		}
	}
	return 0;
}

