#pragma once

#include<fstream>
#include<iostream>
#include<sstream>
#include<numeric>
#include<chrono>
#include<vector>
#include<unordered_map>
#include<opencv2/opencv.hpp>
//#include<dirent.h>
//#include<NvInferRuntime.h>
#include<NvInfer.h>
#include<cuda_runtime_api.h>
#include"logging.h"
#include"BYTETracker.h"


#define CHECK(status) \
    do \
    { \
        auto ret = (status); \
        if (ret != 0) \
        { \
            cerr << "Cuda failure: " << ret << endl; \
            abort(); \
        } \
    } while (0)


using namespace nvinfer1;


enum TASK_MODES
{
	TRACK = 0,
	DETECT = 1
};


enum VISUAL_MODES
{
	ONLINE = 0,
	OFFLINE = 1
};


struct GridAndStride
{
	int grid_x;
	int grid_y;
	int stride;
};


const float color_list[80][3] =
{
	{0.000f, 0.447f, 0.741f},
	{0.850f, 0.325f, 0.098f},
	{0.929f, 0.694f, 0.125f},
	{0.494f, 0.184f, 0.556f},
	{0.466f, 0.674f, 0.188f},
	{0.301f, 0.745f, 0.933f},
	{0.635f, 0.078f, 0.184f},
	{0.300f, 0.300f, 0.300f},
	{0.600f, 0.600f, 0.600f},
	{1.000f, 0.000f, 0.000f},
	{1.000f, 0.500f, 0.000f},
	{0.749f, 0.749f, 0.000f},
	{0.000f, 1.000f, 0.000f},
	{0.000f, 0.000f, 1.000f},
	{0.667f, 0.000f, 1.000f},
	{0.333f, 0.333f, 0.000f},
	{0.333f, 0.667f, 0.000f},
	{0.333f, 1.000f, 0.000f},
	{0.667f, 0.333f, 0.000f},
	{0.667f, 0.667f, 0.000f},
	{0.667f, 1.000f, 0.000f},
	{1.000f, 0.333f, 0.000f},
	{1.000f, 0.667f, 0.000f},
	{1.000f, 1.000f, 0.000f},
	{0.000f, 0.333f, 0.500f},
	{0.000f, 0.667f, 0.500f},
	{0.000f, 1.000f, 0.500f},
	{0.333f, 0.000f, 0.500f},
	{0.333f, 0.333f, 0.500f},
	{0.333f, 0.667f, 0.500f},
	{0.333f, 1.000f, 0.500f},
	{0.667f, 0.000f, 0.500f},
	{0.667f, 0.333f, 0.500f},
	{0.667f, 0.667f, 0.500f},
	{0.667f, 1.000f, 0.500f},
	{1.000f, 0.000f, 0.500f},
	{1.000f, 0.333f, 0.500f},
	{1.000f, 0.667f, 0.500f},
	{1.000f, 1.000f, 0.500f},
	{0.000f, 0.333f, 1.000f},
	{0.000f, 0.667f, 1.000f},
	{0.000f, 1.000f, 1.000f},
	{0.333f, 0.000f, 1.000f},
	{0.333f, 0.333f, 1.000f},
	{0.333f, 0.667f, 1.000f},
	{0.333f, 1.000f, 1.000f},
	{0.667f, 0.000f, 1.000f},
	{0.667f, 0.333f, 1.000f},
	{0.667f, 0.667f, 1.000f},
	{0.667f, 1.000f, 1.000f},
	{1.000f, 0.000f, 1.000f},
	{1.000f, 0.333f, 1.000f},
	{1.000f, 0.667f, 1.000f},
	{0.333f, 0.000f, 0.000f},
	{0.500f, 0.000f, 0.000f},
	{0.667f, 0.000f, 0.000f},
	{0.833f, 0.000f, 0.000f},
	{1.000f, 0.000f, 0.000f},
	{0.000f, 0.167f, 0.000f},
	{0.000f, 0.333f, 0.000f},
	{0.000f, 0.500f, 0.000f},
	{0.000f, 0.667f, 0.000f},
	{0.000f, 0.833f, 0.000f},
	{0.000f, 1.000f, 0.000f},
	{0.000f, 0.000f, 0.167f},
	{0.000f, 0.000f, 0.333f},
	{0.000f, 0.000f, 0.500f},
	{0.000f, 0.000f, 0.667f},
	{0.000f, 0.000f, 0.833f},
	{0.000f, 0.000f, 1.000f},
	{0.000f, 0.000f, 0.000f},
	{0.143f, 0.143f, 0.143f},
	{0.286f, 0.286f, 0.286f},
	{0.429f, 0.429f, 0.429f},
	{0.571f, 0.571f, 0.571f},
	{0.714f, 0.714f, 0.714f},
	{0.857f, 0.857f, 0.857f},
	{0.000f, 0.447f, 0.741f},
	{0.314f, 0.717f, 0.741f},
	{0.500f, 0.500f, 0.000f}
};


void doInference(IExecutionContext& context,
	const int& output_size, const Size& input_shape,
	float* input, float* output);

static void initMappings();

Mat staticResize(const Mat& img);

static void generateGridsAndStride(const int& target_w,
	const int& target_h,
	const vector<int>& strides,
	vector<GridAndStride>& grid_strides);

static inline float intersectionArea(const Object& a, const Object& b)
{
	Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

static void qsortDescentInplace(vector<Object>& faceobjects, int left, int right);

static void qsortDescentInplace(vector<Object>& objects);

static void nmsSortedBboxes(const vector<Object>& faceobjects,
	vector<int>& picked,
	float nms_threshold);

static void generateYoloxProposals(const vector<GridAndStride>& grids_strides,
	const float* output_blob,
	const float& prob_threshold,
	const int& img_w, const int& img_h,
	vector<Object>& objects);

float* blobFromImage(Mat& img);

static void decodeOutputs(const float* output_blob,
	const float& scale,
	const int& img_w,
	const int& img_h,
	vector<Object>& objects);

// ---------- Write tracking resultsS
void drawTrackSC(const std::vector<Track>& output_tracks,
	const int& num_frames, const int& total_ms,
	cv::Mat& img);  // single class

void drawTrackMC(const std::unordered_map<int, vector<Track>>& output_tracks_dict,
	const int& num_frames, const int& total_ms,
	cv::Mat& img);  // multi class


// ---------- Write detection results
void drawDetectMC(const vector<Object>& objects,
	const int& num_frames, const int& total_ms,
	cv::Mat& img);  // multi-class
