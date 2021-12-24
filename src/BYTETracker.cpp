#include"BYTETracker.h"


// ---------- Constructor: do initializations
BYTETracker::BYTETracker(const int& n_classes,
	const int& frame_rate,
	const int& track_buffer,
	const float& high_det_thresh,
	const float& new_track_thresh,
	const float& high_match_thresh,
	const float& low_match_thresh,
	const float& unconfirmed_match_thresh) :
	m_high_det_thresh(high_det_thresh),   // >m_track_thresh as high(1st)
	m_new_track_thresh(new_track_thresh),  // >m_high_thresh as new track
	m_high_match_thresh(high_match_thresh),  // first match threshold
	m_low_match_thresh(low_match_thresh),  // second match threshold
	m_unconfirmed_match_thresh(unconfirmed_match_thresh)  // unconfired match to remain dets
{
	// ----- number of object classes
	this->m_N_CLASSES = n_classes;
	cout << "Total " << n_classes << " classes of object to be tracked.\n";

	this->m_frame_id = 0;
	//this->m_max_time_lost = int(frame_rate / 30.0 * track_buffer);
	this->m_max_time_lost = track_buffer;
	cout << "Max lost time(number of frames): " << m_max_time_lost << endl; 
	cout << "MCMOT tracker inited done" << endl;
}


BYTETracker::~BYTETracker()
{
}


unordered_map<int, vector<Track>> BYTETracker::updateMCMOT(const vector<Object>& objects)
{
	// frame id updating
	this->m_frame_id++;

	// ---------- Track's track id initialization
	if (this->m_frame_id == 1)
	{
		Track::initTrackIDDict(this->m_N_CLASSES);
	}

	// 8 current frame's containers
	unordered_map<int, vector<Track*>> unconfirmed_tracks_dict;
	unordered_map<int, vector<Track*>> tracked_tracks_dict;
	unordered_map<int, vector<Track*>> track_pool_dict;
	unordered_map<int, vector<Track>> activated_tracks_dict;
	unordered_map<int, vector<Track>> refind_tracks_dict;
	unordered_map<int, vector<Track>> lost_tracks_dict;
	unordered_map<int, vector<Track>> removed_tracks_dict;
	unordered_map<int, vector<Track>> output_tracks_dict;

	////////////////// Step 1: Get detections //////////////////
	unordered_map<int, vector<vector<float>>> bboxes_dict;
	unordered_map<int, vector<float>> scores_dict;
	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); ++i)
		{
			const Object& obj = objects[i];

			vector<float> x1y1x2y2;
			x1y1x2y2.resize(4);
			x1y1x2y2[0] = obj.rect.x;					  // x1
			x1y1x2y2[1] = obj.rect.y;					  // y1
			x1y1x2y2[2] = obj.rect.x + obj.rect.width;    // x2
			x1y1x2y2[3] = obj.rect.y + obj.rect.height;   // y2

			const float& score = obj.prob;				  // confidence
			const int& cls_id = obj.label;				  // class ID

			bboxes_dict[cls_id].push_back(x1y1x2y2);
			scores_dict[cls_id].push_back(score);
		}
	}  // non-empty objects assumption

	// ---------- Processing each object classes
	// ----- Build bbox_dict and score_dict
	for (int cls_id = 0; cls_id < this->m_N_CLASSES; ++cls_id)
	{
		// class bboxes
		vector<vector<float>>& cls_bboxes = bboxes_dict[cls_id];

		// class scores
		const vector<float>& cls_scores = scores_dict[cls_id];

		// skip classes of empty detections of objects
		if (cls_bboxes.size() == 0)
		{
			continue;
		}

		// temporary containers
		vector<Track> cls_dets;
		vector<Track> cls_dets_low;
		vector<Track> cls_dets_remain;
		vector<Track> cls_tracked_tracks_tmp;
		vector<Track> cls_res_a, cls_res_b;
		vector<Track*> cls_unmatched_tracks;

		// detections classifications
		for (int i = 0; i < cls_bboxes.size(); ++i)
		{
			vector<float>& tlbr_ = cls_bboxes[i];
			const float& score = cls_scores[i];

			Track track(Track::tlbrTotlwh(tlbr_), score, cls_id);
			if (score > this->m_high_det_thresh)  // high confidence dets
			{
				cls_dets.push_back(track);
			}
			else  // low confidence dets
			{
				cls_dets_low.push_back(track);
			}
		}

		// Add newly detected tracklets to tracked_tracks
		for (int i = 0; i < this->m_tracked_tracks_dict[cls_id].size(); ++i)
		{
			if (!this->m_tracked_tracks_dict[cls_id][i].is_activated)
			{
				unconfirmed_tracks_dict[cls_id].push_back(&this->m_tracked_tracks_dict[cls_id][i]);
			}
			else
			{
				tracked_tracks_dict[cls_id].push_back(&this->m_tracked_tracks_dict[cls_id][i]);
			}
		}

		////////////////// Step 2: First association, with IoU //////////////////
		track_pool_dict[cls_id] = joinTracks(tracked_tracks_dict[cls_id],
			this->m_lost_tracks_dict[cls_id]);
		Track::multiPredict(track_pool_dict[cls_id], this->m_kalman_filter);
		//Track::multiPredict(tracked_tracks_dict[cls_id], this->m_kalman_filter);

		vector<vector<float>> dists;
		int dist_size = 0, dist_size_size = 0;
		dists = iouDistance(track_pool_dict[cls_id], cls_dets, dist_size, dist_size_size);

		vector<vector<int>> matches;
		vector<int> u_track, u_detection;
		linearAssignment(dists, dist_size, dist_size_size, this->m_high_match_thresh,
			matches, u_track, u_detection);

		for (int i = 0; i < matches.size(); ++i)
		{
			Track* track = track_pool_dict[cls_id][matches[i][0]];
			Track* det = &cls_dets[matches[i][1]];
			if (track->state == TrackState::Tracked)
			{
				track->update(*det, this->m_frame_id);
				activated_tracks_dict[cls_id].push_back(*track);
			}
			else
			{
				track->reActivate(*det, this->m_frame_id, false);
				refind_tracks_dict[cls_id].push_back(*track);
			}
		}

		//// ----- Step 3: Second association, using low score dets ----- ////
		for (int i = 0; i < u_detection.size(); ++i)
		{  // store unmatched detections from the the 1st round(high)
			cls_dets_remain.push_back(cls_dets[u_detection[i]]);
		}
		cls_dets.clear();
		cls_dets.assign(cls_dets_low.begin(), cls_dets_low.end());

		// unnatched tacks in track pool to cls_r_tracked_tracks
		for (int i = 0; i < u_track.size(); ++i)
		{
			if (track_pool_dict[cls_id][u_track[i]]->state == TrackState::Tracked)
			{
				cls_unmatched_tracks.push_back(track_pool_dict[cls_id][u_track[i]]);
			}
		}

		dists.clear();
		dists = iouDistance(cls_unmatched_tracks, cls_dets, dist_size, dist_size_size);

		matches.clear();
		u_track.clear();
		u_detection.clear();
		linearAssignment(dists, dist_size, dist_size_size, this->m_low_match_thresh,
			matches, u_track, u_detection);

		for (int i = 0; i < matches.size(); ++i)
		{
			Track* track = cls_unmatched_tracks[matches[i][0]];
			Track* det = &cls_dets[matches[i][1]];
			if (track->state == TrackState::Tracked)
			{
				track->update(*det, this->m_frame_id);
				activated_tracks_dict[cls_id].push_back(*track);
			}
			else
			{
				track->reActivate(*det, this->m_frame_id, false);
				refind_tracks_dict[cls_id].push_back(*track);
			}
		}

		//// @even: process the unmatched dets
		//for (int i = 0; i < u_detection.size(); ++i)
		//{  // store unmatched detections from the the 2nd round(high)
		//	cls_dets_remain.push_back(cls_dets[u_detection[i]]);
		//}

		// process the unmatched tracks for first 2 rounds
		for (int i = 0; i < u_track.size(); ++i)
		{
			Track* track = cls_unmatched_tracks[u_track[i]];
			if (track->state != TrackState::Lost)
			{
				track->markLost();
				lost_tracks_dict[cls_id].push_back(*track);
			}
		}

		// ---------- Deal with unconfirmed tracks, 
		// usually tracks with only one beginning frame
		cls_dets.clear();  // to store unmatched dets in the first round(high)
		cls_dets.assign(cls_dets_remain.begin(), cls_dets_remain.end());

		dists.clear();
		dists = iouDistance(unconfirmed_tracks_dict[cls_id], cls_dets,
			dist_size, dist_size_size);

		matches.clear();
		vector<int> u_unconfirmed;
		u_detection.clear();
		linearAssignment(dists, dist_size, dist_size_size, this->m_unconfirmed_match_thresh,
			matches, u_unconfirmed, u_detection);

		for (int i = 0; i < matches.size(); ++i)
		{
			Track* track = unconfirmed_tracks_dict[cls_id][matches[i][0]];
			Track* det = &cls_dets[matches[i][1]];
			track->update(*det, this->m_frame_id);
			activated_tracks_dict[cls_id].push_back(*track);
		}

		for (int i = 0; i < u_unconfirmed.size(); ++i)
		{
			Track* track = unconfirmed_tracks_dict[cls_id][u_unconfirmed[i]];
			track->markRemoved();
			removed_tracks_dict[cls_id].push_back(*track);
		}

		////////////////// Step 4: Init new tracks //////////////////
		for (int i = 0; i < u_detection.size(); ++i)
		{
			Track* track = &cls_dets[u_detection[i]];
			if (track->score < this->m_new_track_thresh)
			{
				continue;
			}
			track->activate(this->m_kalman_filter, this->m_frame_id);
			activated_tracks_dict[cls_id].push_back(*track);
		}

		////////////////// Step 5: Update state //////////////////
		// ---------- update lost tracks' state
		for (int i = 0; i < this->m_lost_tracks_dict[cls_id].size(); ++i)
		{
			Track& track = this->m_lost_tracks_dict[cls_id][i];
			if (this->m_frame_id - track.endFrame() > this->m_max_time_lost)
			{
				track.markRemoved();
				removed_tracks_dict[cls_id].push_back(track);
			}
		}

		// ---------- Post processing
		// ----- post processing of m_tracked_tracks
		for (int i = 0; i < this->m_tracked_tracks_dict[cls_id].size(); ++i)
		{
			if (this->m_tracked_tracks_dict[cls_id][i].state == TrackState::Tracked)
			{
				cls_tracked_tracks_tmp.push_back(this->m_tracked_tracks_dict[cls_id][i]);
			}
		}
		this->m_tracked_tracks_dict[cls_id].clear();
		this->m_tracked_tracks_dict[cls_id].assign(cls_tracked_tracks_tmp.begin(), cls_tracked_tracks_tmp.end());

		this->m_tracked_tracks_dict[cls_id] = joinTracks(this->m_tracked_tracks_dict[cls_id], activated_tracks_dict[cls_id]);
		this->m_tracked_tracks_dict[cls_id] = joinTracks(this->m_tracked_tracks_dict[cls_id], refind_tracks_dict[cls_id]);

		// ----- post processing of m_lost_tracks
		this->m_lost_tracks_dict[cls_id] = subTracks(this->m_lost_tracks_dict[cls_id],
			this->m_tracked_tracks_dict[cls_id]);
		for (int i = 0; i < lost_tracks_dict[cls_id].size(); ++i)
		{
			this->m_lost_tracks_dict[cls_id].push_back(lost_tracks_dict[cls_id][i]);
		}

		this->m_lost_tracks_dict[cls_id] = subTracks(this->m_lost_tracks_dict[cls_id],
			this->m_removed_tracks_dict[cls_id]);
		for (int i = 0; i < removed_tracks_dict[cls_id].size(); ++i)
		{
			this->m_removed_tracks_dict[cls_id].push_back(removed_tracks_dict[cls_id][i]);
		}

		// remove duplicate
		removeDuplicateTracks(cls_res_a, cls_res_b,
			this->m_tracked_tracks_dict[cls_id], this->m_lost_tracks_dict[cls_id]);

		this->m_tracked_tracks_dict[cls_id].clear();
		this->m_tracked_tracks_dict[cls_id].assign(cls_res_a.begin(), cls_res_a.end());

		this->m_lost_tracks_dict[cls_id].clear();
		this->m_lost_tracks_dict[cls_id].assign(cls_res_b.begin(), cls_res_b.end());

		// return output 
		for (int i = 0; i < this->m_tracked_tracks_dict[cls_id].size(); ++i)
		{
			if (this->m_tracked_tracks_dict[cls_id][i].is_activated)
			{
				output_tracks_dict[cls_id].push_back(this->m_tracked_tracks_dict[cls_id][i]);
			}
		}
	}  // End of class itereations

	return output_tracks_dict;
}

vector<Track> BYTETracker::update(const vector<Object>& objects)
{
	// frame id updating
	this->m_frame_id++;

	// 8 current frame's containers
	vector<Track*> unconfirmed_tracks;
	vector<Track*> tracked_tracks;
	vector<Track*> track_pool;
	vector<Track> activated_tracks;
	vector<Track> refind_tracks;
	vector<Track> lost_tracks;
	vector<Track> removed_tracks;
	vector<Track> output_tracks;

	// tem containers
	vector<Track> detections;
	vector<Track> detections_low;
	vector<Track> detections_cp;
	vector<Track> tracked_tracks_swap;
	vector<Track> res_a, res_b;
	vector<Track*> r_tracked_tracks;

	// ---------- Track's track id initialization
	Track::initTrackIDDict(this->m_N_CLASSES);

	////////////////// Step 1: Get detections //////////////////
	if (objects.size() > 0)
	{
		for (int i = 0; i < objects.size(); i++)
		{
			vector<float> tlbr_;
			tlbr_.resize(4);

			tlbr_[0] = objects[i].rect.x;							// x1
			tlbr_[1] = objects[i].rect.y;							// y1
			tlbr_[2] = objects[i].rect.x + objects[i].rect.width;   // x2
			tlbr_[3] = objects[i].rect.y + objects[i].rect.height;  // y2
			const float& score = objects[i].prob;					// confidence

			Track track(Track::tlbrTotlwh(tlbr_), score, objects[i].label);
			if (score >= this->m_high_det_thresh)
			{
				detections.push_back(track);
			}
			else
			{
				detections_low.push_back(track);
			}
		}
	}

	// Add newly detected tracklets to tracked_tracks
	for (int i = 0; i < this->m_tracked_tracks.size(); i++)
	{
		if (!this->m_tracked_tracks[i].is_activated)
		{
			unconfirmed_tracks.push_back(&this->m_tracked_tracks[i]);
		}
		else
		{
			tracked_tracks.push_back(&this->m_tracked_tracks[i]);
		}
	}

	////////////////// Step 2: First association, with IoU //////////////////
	track_pool = joinTracks(tracked_tracks, this->m_lost_tracks);
	Track::multiPredict(track_pool, this->m_kalman_filter);

	vector<vector<float>> dists;
	int dist_size = 0, dist_size_size = 0;
	dists = iouDistance(track_pool, detections, dist_size, dist_size_size);

	vector<vector<int>> matches;
	vector<int> u_track, u_detection;
	linearAssignment(dists, dist_size, dist_size_size, this->m_high_match_thresh,
		matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		Track* track = track_pool[matches[i][0]];
		Track* det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->m_frame_id);
			activated_tracks.push_back(*track);
		}
		else
		{
			track->reActivate(*det, this->m_frame_id, false);
			refind_tracks.push_back(*track);
		}
	}

	////////////////// Step 3: Second association, using low score dets //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		detections_cp.push_back(detections[u_detection[i]]);
	}
	detections.clear();
	detections.assign(detections_low.begin(), detections_low.end());

	for (int i = 0; i < u_track.size(); i++)
	{
		if (track_pool[u_track[i]]->state == TrackState::Tracked)
		{
			r_tracked_tracks.push_back(track_pool[u_track[i]]);
		}
	}

	dists.clear();
	dists = iouDistance(r_tracked_tracks, detections, dist_size, dist_size_size);

	matches.clear();
	u_track.clear();
	u_detection.clear();
	linearAssignment(dists, dist_size, dist_size_size, 0.5, matches, u_track, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		Track* track = r_tracked_tracks[matches[i][0]];
		Track* det = &detections[matches[i][1]];
		if (track->state == TrackState::Tracked)
		{
			track->update(*det, this->m_frame_id);
			activated_tracks.push_back(*track);
		}
		else
		{
			track->reActivate(*det, this->m_frame_id, false);
			refind_tracks.push_back(*track);
		}
	}

	for (int i = 0; i < u_track.size(); i++)
	{
		Track* track = r_tracked_tracks[u_track[i]];
		if (track->state != TrackState::Lost)
		{
			track->markLost();
			lost_tracks.push_back(*track);
		}
	}

	// Deal with unconfirmed tracks, usually tracks with only one beginning frame
	detections.clear();
	detections.assign(detections_cp.begin(), detections_cp.end());

	dists.clear();
	dists = iouDistance(unconfirmed_tracks, detections, dist_size, dist_size_size);

	matches.clear();
	vector<int> u_unconfirmed;
	u_detection.clear();
	linearAssignment(dists, dist_size, dist_size_size, 0.7, matches, u_unconfirmed, u_detection);

	for (int i = 0; i < matches.size(); i++)
	{
		unconfirmed_tracks[matches[i][0]]->update(detections[matches[i][1]], this->m_frame_id);
		activated_tracks.push_back(*unconfirmed_tracks[matches[i][0]]);
	}

	for (int i = 0; i < u_unconfirmed.size(); i++)
	{
		Track* track = unconfirmed_tracks[u_unconfirmed[i]];
		track->markRemoved();
		removed_tracks.push_back(*track);
	}

	////////////////// Step 4: Init new tracks //////////////////
	for (int i = 0; i < u_detection.size(); i++)
	{
		Track* track = &detections[u_detection[i]];
		if (track->score < this->m_new_track_thresh)
		{
			continue;
		}
		track->activate(this->m_kalman_filter, this->m_frame_id);
		activated_tracks.push_back(*track);
	}

	////////////////// Step 5: Update state //////////////////
	for (int i = 0; i < this->m_lost_tracks.size(); i++)
	{
		if (this->m_frame_id - this->m_lost_tracks[i].endFrame() > this->m_max_time_lost)
		{
			this->m_lost_tracks[i].markRemoved();
			removed_tracks.push_back(this->m_lost_tracks[i]);
		}
	}

	for (int i = 0; i < this->m_tracked_tracks.size(); i++)
	{
		if (this->m_tracked_tracks[i].state == TrackState::Tracked)
		{
			tracked_tracks_swap.push_back(this->m_tracked_tracks[i]);
		}
	}
	this->m_tracked_tracks.clear();
	this->m_tracked_tracks.assign(tracked_tracks_swap.begin(), tracked_tracks_swap.end());

	this->m_tracked_tracks = joinTracks(this->m_tracked_tracks, activated_tracks);
	this->m_tracked_tracks = joinTracks(this->m_tracked_tracks, refind_tracks);

	//std::cout << activated_tracks.size() << std::endl;

	this->m_lost_tracks = subTracks(this->m_lost_tracks, this->m_tracked_tracks);
	for (int i = 0; i < lost_tracks.size(); i++)
	{
		this->m_lost_tracks.push_back(lost_tracks[i]);
	}

	this->m_lost_tracks = subTracks(this->m_lost_tracks, this->m_removed_tracks);
	for (int i = 0; i < removed_tracks.size(); i++)
	{
		this->m_removed_tracks.push_back(removed_tracks[i]);
	}

	removeDuplicateTracks(res_a, res_b, this->m_tracked_tracks, this->m_lost_tracks);

	this->m_tracked_tracks.clear();
	this->m_tracked_tracks.assign(res_a.begin(), res_a.end());
	this->m_lost_tracks.clear();
	this->m_lost_tracks.assign(res_b.begin(), res_b.end());

	for (int i = 0; i < this->m_tracked_tracks.size(); i++)
	{
		if (this->m_tracked_tracks[i].is_activated)
		{
			output_tracks.push_back(this->m_tracked_tracks[i]);
		}
	}

	return output_tracks;
}


