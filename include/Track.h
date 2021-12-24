#pragma once

#include<opencv2/opencv.hpp>
#include<unordered_map>
#include"kalmanFilter.h"


using namespace cv;
using namespace std;


enum TrackState 
{
	New = 0, 
	Tracked, 
	Lost, 
	Removed 
};


class Track
{
public:
	Track(const vector<float>& tlwh_, const float& score, const int& cls_id);
	~Track();

	vector<float> static tlbrTotlwh(vector<float>& tlbr);

	void static multiPredict(vector<Track*>& tracks,
		byte_kalman::KalmanFilter& kalman_filter);

	void staticTLWH();

	void staticTLBR();

	vector<float> tlwhToxyah(vector<float> tlwh_tmp);

	vector<float> toXYAH();

	void markLost();

	void markRemoved();

	static void initTrackIDDict(const int n_classes);

	static int nextID(const int& cls_id);

	//static void printIDDict();

	int endFrame();
	
	void activate(byte_kalman::KalmanFilter& kalman_filter, int frame_id);

	void reActivate(Track& new_track, int frame_id, bool new_id = false);

	void update(Track& new_track, int frame_id);

public:
	bool is_activated;  // top tracking state
	int track_id;
	int class_id;
	int state;

	vector<float> _tlwh;
	vector<float> tlwh;  // x1y1wh
	vector<float> tlbr;  // x1y1x2y2
	int frame_id;
	int tracklet_len;
	int start_frame;

	KAL_MEAN mean;
	KAL_COVA covariance;
	float score;

	// mapping each class id to the track id count of this class
	static unordered_map<int, int> static_track_id_dict;  // 类静态成员变量声明

private:
	byte_kalman::KalmanFilter kalman_filter;
};