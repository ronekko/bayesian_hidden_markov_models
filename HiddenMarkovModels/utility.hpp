#pragma once

#include "stdafx.h"

namespace util
{

using namespace std;

template<typename T>
void print(vector<vector<T>> x){	
	for(auto xk:x){
		for(int i=0; i<xk.size(); ++i){
			cout << xk[i] << " ";
		}
		cout << endl;
	}cout << endl;
}

template<typename T>
void print(vector<T> x){	
	for(auto xk:x){
		cout << xk << " ";
	}cout << endl;
}

// 画像サイズをscale倍に拡大する。元画像の1ピクセルが、拡大後の画像で一辺scale [pixel]の正方形となるように拡大する。つまり平滑化しない。
inline cv::Mat upsample(const cv::Mat &image, const int &scale)
{
	int w = scale * image.cols;
	int h = scale * image.rows;
	int depth = image.depth();
	int channels = image.channels();
	cv::Mat result(h, w, CV_MAKETYPE(depth, channels));

	for(int y=0; y<h; ++y){
		for(int x=0; x<w; ++x){
			for(int c=0; c<channels; ++c){
				int xx = x / scale;
				int yy = y / scale;
				int ww = w / scale;
				result.data[(y * w + x) * channels + c] = image.data[(yy * ww + xx) * channels + c];
			}
		}
	}
	return result;
}



// shapeかscaleがほぼ0のときは常に0を返す（ガンマ分布の定義は 0<shape, 0<scale であるため）
inline double gammaRandom(boost::mt19937 &engine, const double &shape, const double &scale)
{
	if(shape < 1.0e-300 || scale < 1.0e-300){ return 0; }
	boost::math::gamma_distribution<> dist(shape, scale);
	return boost::math::quantile(dist, boost::uniform_01<>()(engine));
}


// ディリクレ分布から乱数を生成する http://en.wikipedia.org/wiki/Dirichlet_distribution#Gamma_distribution
inline std::vector<double> dirichletRandom(boost::mt19937 &engine, const std::vector<double> &alpha)
{	
	const int K = alpha.size();
	std::vector<double> y(K);
	double sumY = 0.0;

	for(int k=0; k<K; ++k){
		//y[k] = boost::gamma_distribution<>(alpha[k], 1.0)(engine);	// shapeパラメータが大きいと落ちる
		y[k] = gammaRandom(engine, alpha[k], 1.0);
		sumY += y[k];
	}

	for(int k=0; k<K; ++k){
		y[k] /= sumY;
	}

	return y;
}


inline double betaRandom(boost::mt19937 &engine, const double &alpha, const double &beta)
{
	boost::math::beta_distribution<> dist(alpha, beta);
	return boost::math::quantile(dist, boost::uniform_01<>()(engine));
}

// 多項分布からのサンプリング、ただしパラメータは正規化されていない（\sum p_iが1とは限らない）
inline int multinomialByUnnormalizedParameters(boost::mt19937 &engine, const vector<double> &p)
{
	const int K=p.size();
	vector<double> CDF(K);
	double z = 0.0;
	for(int k=0; k<K; ++k){
		CDF[k] = z + p[k];
		z = CDF[k];
	}

	double u = boost::uniform_01<>()(engine) * CDF.back();
	for(int k=0; k<K; ++k){
		if(u < CDF[k]){
			return k;
		}
	}
	cout <<"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
	return K-1;
}

inline double logsumexp (double x, double y, bool flg)
{
	if (flg) return y; // init mode
	if (x == y) return x + 0.69314718055; // log(2)
	double vmin = std::min (x, y);
	double vmax = std::max (x, y);
	if (vmax > vmin + 50) {
		return vmax;
	} else {
		return vmax + std::log (std::exp (vmin - vmax) + 1.0);
	}
}
// 多項分布からのサンプリング、ただしパラメータはlog(p_1), ... , log(p_K)で与えられ、正規化されていない（\sum p_iが1とは限らない）
inline int multinomialByUnnormalizedLogParameters(boost::mt19937 &engine, const vector<double> &lnp)
{
	const int K=lnp.size();
	vector<double> logCDF(K);
	double z = 0.0;
	for(int k=0; k<K; ++k){
		z = logsumexp(z, lnp[k], (k==0));
		logCDF[k] = z;
	}

	double u = log(boost::uniform_01<>()(engine)) + logCDF.back();
	for(int k=0; k<K; ++k){
		if(u < logCDF[k]){
			return k;
		}
	}

	return K-1;
}

// Chinese restaurant table distributionからの乱数生成
inline int CRTRandom(boost::mt19937 &engine, const int &n, const double &alpha)
{
	if(n < 1){ return 0; }
	if(alpha == 0){ return 1;}

	int l=1;
	for(int i=1; i<n; ++i){
		double p = alpha / (i + alpha);
		l += boost::bernoulli_distribution<>(p)(engine);
	}
	return l;
}

inline void test_CRTRandom(void)
{
	boost::mt19937 engine(0);
	double alpha = 4.0;
	int r = 100000000;
	{
		//int n = 5; double pmf[] = {0, 24, 50, 35, 10, 1};
		//int n = 6; double pmf[] = {0, 120, 274, 225, 85, 15, 1};
		//int n = 7; double pmf[] = {0, 720, 1764, 1624, 735, 175, 21, 1};
		//int n = 8; double pmf[] = {0, 5040, 13068, 13132, 6769, 1960, 322, 28, 1};
		int n = 9; double pmf[] = {0, 40320, 109584, 118124, 67284, 22449, 4536, 546, 36, 1};
		for(int l=1; l<=n; ++l){	pmf[l] *= pow(alpha, l); }
		double total = boost::accumulate(pmf, 0.0);
		for(int l=1; l<=n; ++l){	pmf[l] /= total; }

		vector<int> result(n+1, 0);
		for(int i=0; i<r; ++i){
			int l = CRTRandom(engine, n, alpha);
			result[l]++;
		}

		vector<double> prop(n+1, 0.0);
		for(int l=1; l<=n; ++l){
			prop[l] = result[l] / static_cast<double>(r);
		}
		
		cout << "empirical\tpmf[l]\t\tdiff" << endl;
		for(int l=0; l<=n; ++l){
			cout << prop[l] << "    \t" << pmf[l] << "    \t" << prop[l] - pmf[l] << endl;
		}
	}
}

//! Write cv::Mat as binary
/*!
\param[out] ofs output file stream
\param[in] out_mat mat to save
*/
inline bool writeMatBinary(std::ofstream& ofs, const cv::Mat& out_mat)
{
	if(!ofs.is_open()){
		return false;
	}
	if(out_mat.empty()){
		int s = 0;
		ofs.write((const char*)(&s), sizeof(int));
		return true;
	}
	int type = out_mat.type();
	ofs.write((const char*)(&out_mat.rows), sizeof(int));
	ofs.write((const char*)(&out_mat.cols), sizeof(int));
	ofs.write((const char*)(&type), sizeof(int));
	ofs.write((const char*)(out_mat.data), out_mat.elemSize() * out_mat.total());

	return true;
}


//! Save cv::Mat as binary
/*!
\param[in] filename filaname to save
\param[in] output cvmat to save
*/
inline bool saveMatBinary(const std::string& filename, const cv::Mat& output){
	std::ofstream ofs(filename, std::ios::binary);
	return writeMatBinary(ofs, output);
}


//! Read cv::Mat from binary
/*!
\param[in] ifs input file stream
\param[out] in_mat mat to load
*/
inline bool readMatBinary(std::ifstream& ifs, cv::Mat& in_mat)
{
	if(!ifs.is_open()){
		return false;
	}

	int rows, cols, type;
		ifs.read((char*)(&rows), sizeof(int));
		if(rows==0){
		return true;
	}
	ifs.read((char*)(&cols), sizeof(int));
	ifs.read((char*)(&type), sizeof(int));

	in_mat.release();
	in_mat.create(rows, cols, type);
	ifs.read((char*)(in_mat.data), in_mat.elemSize() * in_mat.total());

	return true;
}


//! Load cv::Mat as binary
/*!
\param[in] filename filaname to load
\param[out] output loaded cv::Mat
*/
inline bool loadMatBinary(const std::string& filename, cv::Mat& output){
	std::ifstream ifs(filename, std::ios::binary);
	return readMatBinary(ifs, output);
}

inline void save_pca(const cv::PCA &pca, const string &filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << "pca_mean" << pca.mean;
	fs << "pca_eigenvalues" << pca.eigenvalues;
	fs << "pca_eivenvectors" << pca.eigenvectors;
}

inline cv::PCA load_pca(const string &filename)
{
	cv::PCA pca;
	cv::FileStorage fs(filename, cv::FileStorage::READ);
	fs["pca_mean"] >> pca.mean;
	fs["pca_eigenvalues"] >> pca.eigenvalues;
	fs["pca_eivenvectors"] >> pca.eigenvectors;
	return pca;
}

template<typename _EM>
inline void save_em(const _EM &gmm, const string &filename)
{
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << "covMatType" << gmm.get<int>("covMatType");
	fs << "covs" << gmm.get<vector<cv::Mat>>("covs");
	fs << "epsilon" << gmm.get<double>("epsilon");
	fs << "maxIters" << gmm.get<int>("maxIters");
	fs << "means" << gmm.get<cv::Mat>("means");
	fs << "nclusters" << gmm.get<int>("nclusters");
	fs << "weights" << gmm.get<cv::Mat>("weights");
}

// Unsigned Stirling numbers of the 1st kind
// Typically, this function might return large value.
// For example, max n is 16 in int, 170 in double.
// so if n is greater than 170, one must use boost::multiprecision::cpp_dec_float_100 FLOAT
template<typename NUMERIC_TYPE>
NUMERIC_TYPE u_stirling_1st(int n, int k)
{
	static vector<vector<NUMERIC_TYPE>> s(1, vector<NUMERIC_TYPE>(1,1));
	int N = s.size();
	if(n >= N){
		for(int i=N; i<=n; ++i){
			s.push_back(vector<NUMERIC_TYPE>(i+1, 0));
			for(int j=1; j<i; ++j){
				s[i][j] = s[i-1][j-1] + (i -1) * s[i-1][j];
			}
			s[i][i] = 1;
		}
	}
	return s[n][k];
}

// vector<double>型などをL1正規化する
template<typename T>
T l1_normalize(const T &v){
	T::value_type sum = boost::accumulate(v, static_cast<T::value_type>(0));
	T ret(v.begin(), v.end());
	for(auto &ret_i: ret){
		ret_i /= sum;
	}
	return ret;
}

// vector<double>型などをL2正規化する
template<typename T>
T l2_normalize(const T &v){
	T::value_type sum = static_cast<T::value_type>(0);
	for(const auto &v_i: v){
		sum += v_i * v_i;
	}
	sum = sqrt(sum);
	
	T ret(v.begin(), v.end());
	for(auto &ret_i: ret){
		ret_i /= sum;
	}
	return ret;
}
};


namespace cv{
/****************************************************************************************\
*                             ColorDescriptorExtractor                           *
\****************************************************************************************/
class CV_EXPORTS ColorDescriptorExtractor : public cv::DescriptorExtractor
{
private:
	cv::Ptr<DescriptorExtractor> descriptorExtractor;
	int cvt_color_code;
	static const int COLOR_BGR_WITHOUT_CHANGE = -1;

public:
	ColorDescriptorExtractor( const cv::Ptr<DescriptorExtractor>& _descriptorExtractor) :
			descriptorExtractor(_descriptorExtractor),
			cvt_color_code(-1)
	{
		CV_Assert( !descriptorExtractor.empty() );
	}

	void setColorCode(const int _cvt_color_code){
		cvt_color_code = _cvt_color_code;
	}

	struct KP_LessThan
	{
		KP_LessThan(const vector<cv::KeyPoint>& _kp) : kp(&_kp) {}
		bool operator()(int i, int j) const
		{
			return (*kp)[i].class_id < (*kp)[j].class_id;
		}
		const vector<cv::KeyPoint>* kp;
	};

	// 3チャネル画像をopponent color spaceに変換せずそのまま各チャネルでディスクリプタを求めて結合する
	// L*a*b表現の画像などについて用いる (L*a*bをさらにopponent変換するのは不適切だから)
	void computeImpl( const cv::Mat& bgrImage, vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors )const {
		vector<cv::Mat> newChannels;
		convertBGRImageToAnotherColorSpace( bgrImage, newChannels, cvt_color_code );

		const int N = 3; // channels count
		vector<cv::KeyPoint> channelKeypoints[N];
		cv::Mat channelDescriptors[N];
		vector<int> idxs[N];

		// Compute descriptors three times, once for each Opponent channel to concatenate into a single color descriptor
		int maxKeypointsCount = 0;
		for( int ci = 0; ci < N; ci++ )
		{
			channelKeypoints[ci].insert( channelKeypoints[ci].begin(), keypoints.begin(), keypoints.end() );
			// Use class_id member to get indices into initial keypoints vector
			for( size_t ki = 0; ki < channelKeypoints[ci].size(); ki++ )
				channelKeypoints[ci][ki].class_id = (int)ki;

			descriptorExtractor->compute( newChannels[ci], channelKeypoints[ci], channelDescriptors[ci] );
			idxs[ci].resize( channelKeypoints[ci].size() );
			for( size_t ki = 0; ki < channelKeypoints[ci].size(); ki++ )
			{
				idxs[ci][ki] = (int)ki;
			}
			std::sort( idxs[ci].begin(), idxs[ci].end(), KP_LessThan(channelKeypoints[ci]) );
			maxKeypointsCount = std::max( maxKeypointsCount, (int)channelKeypoints[ci].size());
		}

		vector<cv::KeyPoint> outKeypoints;
		outKeypoints.reserve( keypoints.size() );

		int dSize = descriptorExtractor->descriptorSize();
		cv::Mat mergedDescriptors( maxKeypointsCount, 3*dSize, descriptorExtractor->descriptorType() );
		int mergedCount = 0;
		// cp - current channel position
		size_t cp[] = {0, 0, 0};
		while( cp[0] < channelKeypoints[0].size() &&
			   cp[1] < channelKeypoints[1].size() &&
			   cp[2] < channelKeypoints[2].size() )
		{
			const int maxInitIdx = std::max( 0, std::max( channelKeypoints[0][idxs[0][cp[0]]].class_id,
														  std::max( channelKeypoints[1][idxs[1][cp[1]]].class_id,
																	channelKeypoints[2][idxs[2][cp[2]]].class_id ) ) );

			while( channelKeypoints[0][idxs[0][cp[0]]].class_id < maxInitIdx && cp[0] < channelKeypoints[0].size() ) { cp[0]++; }
			while( channelKeypoints[1][idxs[1][cp[1]]].class_id < maxInitIdx && cp[1] < channelKeypoints[1].size() ) { cp[1]++; }
			while( channelKeypoints[2][idxs[2][cp[2]]].class_id < maxInitIdx && cp[2] < channelKeypoints[2].size() ) { cp[2]++; }
			if( cp[0] >= channelKeypoints[0].size() || cp[1] >= channelKeypoints[1].size() || cp[2] >= channelKeypoints[2].size() )
				break;

			if( channelKeypoints[0][idxs[0][cp[0]]].class_id == maxInitIdx &&
				channelKeypoints[1][idxs[1][cp[1]]].class_id == maxInitIdx &&
				channelKeypoints[2][idxs[2][cp[2]]].class_id == maxInitIdx )
			{
				outKeypoints.push_back( keypoints[maxInitIdx] );
				// merge descriptors
				for( int ci = 0; ci < N; ci++ )
				{
					cv::Mat dst = mergedDescriptors(cv::Range(mergedCount, mergedCount+1), cv::Range(ci*dSize, (ci+1)*dSize));
					channelDescriptors[ci].row( idxs[ci][cp[ci]] ).copyTo( dst );
					cp[ci]++;
				}
				mergedCount++;
			}
		}
		mergedDescriptors.rowRange(0, mergedCount).copyTo( descriptors );
		std::swap( outKeypoints, keypoints );
	};


	inline static void convertBGRImageToAnotherColorSpace( const cv::Mat& bgrImage, vector<cv::Mat>& newChannels, int cvt_color_code )
	{
		const int COLOR_BGR_WITHOUT_CHANGE = -1;
		if( bgrImage.type() != CV_8UC3 )
			CV_Error( CV_StsBadArg, "input image must be an BGR image of type CV_8UC3" );

		if(cvt_color_code != COLOR_BGR_WITHOUT_CHANGE){
			cvtColor(bgrImage, bgrImage, cvt_color_code);
		}
		split(bgrImage, newChannels);
	}



	void read( const cv::FileNode& fn )
	{
		descriptorExtractor->read(fn);
	}

	void write( cv::FileStorage& fs ) const
	{
		descriptorExtractor->write(fs);
	}

	int descriptorSize() const
	{
		return 3*descriptorExtractor->descriptorSize();
	}

	int descriptorType() const
	{
		return descriptorExtractor->descriptorType();
	}

	bool empty() const
	{
		return descriptorExtractor.empty() || (DescriptorExtractor*)(descriptorExtractor)->empty();
	}
};
};
