#include"Track.h"


unordered_map<int, int> Track::static_track_id_dict;  // 类静态成员变量定义


Track::Track(const vector<float>& tlwh_, const float& score, const int& cls_id)
{
	_tlwh.resize(4);
	_tlwh.assign(tlwh_.begin(), tlwh_.end());

	is_activated = false;  // init activate flag to be false
	track_id = 0;
	state = TrackState::New;
	
	tlwh.resize(4);
	tlbr.resize(4);

	staticTLWH();
	staticTLBR();

	frame_id = 0;
	tracklet_len = 0;

	this->class_id = cls_id;  // object class id
	this->score = score;

	start_frame = 0;
}


Track::~Track()
{
}


void Track::initTrackIDDict(const int n_classes)
{
	for (int i = 0; i < 5; ++i)
	{
		Track::static_track_id_dict[i] = 0;
	}
}


void Track::activate(byte_kalman::KalmanFilter& kalman_filter, int frame_id)
{
	this->kalman_filter = kalman_filter;  // send the shared kalman filter
	this->track_id = Track::nextID(this->class_id);

	vector<float> _tlwh_tmp(4);
	_tlwh_tmp[0] = this->_tlwh[0];
	_tlwh_tmp[1] = this->_tlwh[1];
	_tlwh_tmp[2] = this->_tlwh[2];
	_tlwh_tmp[3] = this->_tlwh[3];
	vector<float> xyah = tlwhToxyah(_tlwh_tmp);
	DETECT_BOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.initiate(xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	staticTLWH();
	staticTLBR();

	this->tracklet_len = 0;

	this->state = TrackState::Tracked;
	if (frame_id == 1)
	{
		this->is_activated = true;
	}

	//this->is_activated = true;
	this->frame_id = frame_id;

	// set start frame
	this->start_frame = frame_id;  
}


void Track::reActivate(Track& new_track, int frame_id, bool new_id)
{
	vector<float> xyah = tlwhToxyah(new_track.tlwh);

	DETECT_BOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];
	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);

	this->mean = mc.first;
	this->covariance = mc.second;

	staticTLWH();
	staticTLBR();

	this->tracklet_len = 0;
	this->frame_id = frame_id;
	this->score = new_track.score;

	this->state = TrackState::Tracked;
	this->is_activated = true;  // set to be activated

	if (new_id)
	{
		this->track_id = Track::nextID(this->class_id);
	}
}


void Track::update(Track& new_track, int frame_id)
{
	this->frame_id = frame_id;
	this->tracklet_len++;

	vector<float> xyah = tlwhToxyah(new_track.tlwh);
	DETECT_BOX xyah_box;
	xyah_box[0] = xyah[0];
	xyah_box[1] = xyah[1];
	xyah_box[2] = xyah[2];
	xyah_box[3] = xyah[3];

	auto mc = this->kalman_filter.update(this->mean, this->covariance, xyah_box);
	this->mean = mc.first;
	this->covariance = mc.second;

	staticTLWH();
	staticTLBR();

	this->state = TrackState::Tracked;
	this->is_activated = true;  // set to be activated

	this->score = new_track.score;
}


void Track::staticTLWH()
{
	if (this->state == TrackState::New)
	{
		tlwh[0] = _tlwh[0];
		tlwh[1] = _tlwh[1];
		tlwh[2] = _tlwh[2];
		tlwh[3] = _tlwh[3];
		return;
	}

	tlwh[0] = mean[0];  // x(center_x)
	tlwh[1] = mean[1];  // y(center_y)
	tlwh[2] = mean[2];  // a(a=w/h, aspect ratio)
	tlwh[3] = mean[3];  // h

	tlwh[2] *= tlwh[3];  // -> center_x, center_y, w, h

	// -> x1y1wh
	tlwh[0] -= tlwh[2] * 0.5f;
	tlwh[1] -= tlwh[3] * 0.5f;
}


void Track::staticTLBR()
{// x1y1wh -> x1y1x2y2
	tlbr.clear();
	tlbr.assign(tlwh.begin(), tlwh.end());
	tlbr[2] += tlbr[0];
	tlbr[3] += tlbr[1];
}


vector<float> Track::tlwhToxyah(vector<float> tlwh_tmp)
{
	vector<float> tlwh_output = tlwh_tmp;
	tlwh_output[0] += tlwh_output[2] * 0.5f;
	tlwh_output[1] += tlwh_output[3] * 0.5f;
	tlwh_output[2] /= tlwh_output[3];
	return tlwh_output;
}


vector<float> Track::toXYAH()
{
	return tlwhToxyah(tlwh);
}


vector<float> Track::tlbrTotlwh(vector<float>& tlbr)
{// x1y1x2y2 -> x1y1wh
	tlbr[2] -= tlbr[0];  // w = x2 - x1
	tlbr[3] -= tlbr[1];
	return tlbr;
}


void Track::markLost()
{
	state = TrackState::Lost;
}


void Track::markRemoved()
{
	state = TrackState::Removed;
}


int Track::nextID(const int& cls_id)
{
	Track::static_track_id_dict[cls_id] += 1;
	return Track::static_track_id_dict[cls_id];
}

//void Track::printIDDict()
//{
//	for (auto it = Track::static_track_id_dict.begin();
//		it != Track::static_track_id_dict.end(); ++it)
//	{
//		cout << it->first << ": " << it->second << endl;
//	}
//}

int Track::endFrame()
{
	return this->frame_id;
}


void Track::multiPredict(vector<Track*>& tracks, byte_kalman::KalmanFilter& kalman_filter)
{
	for (int i = 0; i < tracks.size(); ++i)
	{
		if (tracks[i]->state != TrackState::Tracked)
		{
			tracks[i]->mean[7] = 0;
		}
		kalman_filter.predict(tracks[i]->mean, tracks[i]->covariance);
	}
}


//int Track::nextID()
//{
//	static int _count = 0;
//	_count++;
//	return _count;
//}