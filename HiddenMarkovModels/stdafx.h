// stdafx.h : 標準のシステム インクルード ファイルのインクルード ファイル、または
// 参照回数が多く、かつあまり変更されない、プロジェクト専用のインクルード ファイル
// を記述します。
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>



// TODO: プログラムに必要な追加ヘッダーをここで参照してください。
#ifdef _DEBUG
	#pragma comment(lib, "opencv_core246d.lib")
	#pragma comment(lib, "opencv_imgproc246d.lib")
	#pragma comment(lib, "opencv_highgui246d.lib")
	#pragma comment(lib, "opencv_features2d246d.lib")
	#pragma comment(lib, "opencv_nonfree246d.lib")
	#pragma comment(lib, "opencv_ml246d.lib")
#else
	#pragma comment(lib, "opencv_core246.lib")
	#pragma comment(lib, "opencv_imgproc246.lib")
	#pragma comment(lib, "opencv_highgui246.lib")
	#pragma comment(lib, "opencv_features2d246.lib")
	#pragma comment(lib, "opencv_nonfree246.lib")
	#pragma comment(lib, "opencv_ml246.lib")
#endif


#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <utility>
#include <memory>
#include <typeinfo>

#include <boost/range/algorithm.hpp>
#include <boost/range/algorithm_ext.hpp>
#include <boost/range/numeric.hpp>
#include <boost/range/irange.hpp>
#include <boost/timer.hpp>
#include <boost/math/distributions.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions.hpp>
#include <boost/random.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>  

#include <opencv2/opencv.hpp>
