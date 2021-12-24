#pragma once

#include"Track.h"


struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
};


class BYTETracker
{
public:
	BYTETracker(const int& n_classes,
		const int& frame_rate,
		const int& track_buffer,
		const float& high_det_thresh,
		const float& new_track_thresh,
		const float& high_match_thresh,
		const float& low_match_thresh,
		const float& unconfirmed_match_thresh);
	~BYTETracker();

	vector<Track> update(const vector<Object>& objects);
	unordered_map<int, vector<Track>> updateMCMOT(const vector<Object>& objects);

	static Scalar getColor(int idx);

private:
	vector<Track*> joinTracks(vector<Track*>& tlista,
		vector<Track>& tlistb);

	vector<Track> joinTracks(vector<Track>& tlista,
		vector<Track>& tlistb);

	vector<Track> subTracks(vector<Track>& tlista,
		vector<Track>& tlistb);

	void removeDuplicateTracks(vector<Track>& resa,
		vector<Track>& resb,
		vector<Track>& tracks_a,
		vector<Track>& tracks_b);

	void linearAssignment(vector<vector<float>>& cost_matrix, 
		int cost_matrix_size, 
		int cost_matrix_size_size,
		float thresh,
		vector<vector<int>>& matches,
		vector<int>& unmatched_a,
		vector<int>& unmatched_b);

	vector<vector<float>> iouDistance(vector<Track*>& atracks,
		vector<Track>& btracks, 
		int& dist_size,
		int& dist_size_size);

	vector<vector<float>> iouDistance(vector<Track>& atracks, 
		vector<Track>& btracks);

	vector<vector<float>> ious(vector<vector<float>>& atlbrs,
		vector<vector<float>>& btlbrs);

	double lapjv(const vector<vector<float>>& cost,
		vector<int>& rowsol, 
		vector<int>& colsol,
		bool extend_cost = false, 
		float cost_limit = LONG_MAX, 
		bool return_cost = true);

private:

	float m_high_det_thresh;
	float m_new_track_thresh;
	float m_high_match_thresh;
	float m_low_match_thresh;
	float m_unconfirmed_match_thresh;
	int m_frame_id;
	int m_max_time_lost;

	// tracking object class number
	int m_N_CLASSES;

	// 3 containers of the tracker
	vector<Track> m_tracked_tracks;
	vector<Track> m_lost_tracks;
	vector<Track> m_removed_tracks;

	unordered_map<int, vector<Track>> m_tracked_tracks_dict;
	unordered_map<int, vector<Track>> m_lost_tracks_dict;
	unordered_map<int, vector<Track>> m_removed_tracks_dict;

	byte_kalman::KalmanFilter m_kalman_filter;
};