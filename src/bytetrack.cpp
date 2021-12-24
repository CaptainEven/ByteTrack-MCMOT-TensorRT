#include "bytetrack.h"


// ---------- Hyper parameters definition
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.7
#define BBOX_CONF_THRESH 0.1  // 0.1


const int NUM_CLASSES(5);  // number of object classes
const int TRACK_BUFFER(240);  // frame number of tracking states buffers
const int TASK(TASK_MODES::TRACK);  // task mode: TRACK | DETECT
const int VISUAL(VISUAL_MODES::ONLINE); // visualization mode: ONLINE | OFFLINE
const float RESIZE_RATIO(0.55f);  // Resizing rario to visualize online
const int DELAY(1);  // online frame visualization delay(ms)

// 2 Det/Track classification thresholds for the byte tracker
const float HIGH_DET_THRESH(0.5f);  // 0.5f > m_track_thresh as high(1st)
const float NEW_TRACK_THRESH(HIGH_DET_THRESH + 0.1f);   // > m_high_thresh as new track

// 3 Matching thresholds
const float HIGH_MATCH_THRESH(0.8f);  // 0.8f first match threshold
const float LOW_MATCH_THRESH(0.5f);  // 0.5f second match threshold
const float UNCONFIRMED_MATCH_THRESH(0.7f);  // 0.7: unconfirmed track match to remain dets


// ---------- stuff we know about the network and the input/output blobs
static const int NET_W(768);  // 1088
static const int NET_H(448);  // 608
const char* INPUT_BLOB_NAME = "input_0";
const char* OUTPUT_BLOB_NAME = "output_0";

// ----- Define the logger
static sample::Logger gLogger;

static vector<string> CLASSES({
	"car",
	"bicycle",
	"person",
	"cyclist",
	"tricycle"
	});
static unordered_map<string, int> CLASS2ID;
static unordered_map<int, string> ID2CLASS;

// ---------- Define file paths
static const string VIDEO_PATH("../videos/7.mp4");
static const string ENGINE_PATH("./engines/yolox_tiny_det_c5_trt_f16.engine");


/**********Entrance(main)**********/
int main(int argc, char** argv)
{
	cudaSetDevice(DEVICE);

	// create a model using the API directly and serialize it to a stream
	char* TRT_Model_Stream(nullptr);
	size_t size(0);

	// ---------- Open engine and video file
	ifstream file(ENGINE_PATH, ios::binary);
	if (file.good())
	{
		file.seekg(0, file.end);
		size = file.tellg();
		file.seekg(0, file.beg);
		TRT_Model_Stream = new char[size];
		assert(TRT_Model_Stream);
		file.read(TRT_Model_Stream, size);
		file.close();
		cout << "[I] " << ENGINE_PATH << " loaded." << endl;
	}
	else
	{
		cout << "[E] read engine file " << ENGINE_PATH << " failed!" << endl;
	}

	if (TASK == TASK_MODES::TRACK)
	{
		cout << "[I] Task mode: Track" << endl;
	}
	else if (TASK == TASK_MODES::DETECT)
	{
		cout << "[I] Task mode: Detect" << endl;
	}
	if (VISUAL == VISUAL_MODES::ONLINE)
	{
		cout << "[I] Visualization mode: Online" << endl;
	}
	else if (VISUAL == VISUAL_MODES::OFFLINE)
	{
		cout << "[I] Visualization mode: Offline" << endl;
	}

	const string input_video_path(VIDEO_PATH);

	IRuntime* runtime = createInferRuntime(gLogger.getTRTLogger());
	assert(runtime != nullptr);

	ICudaEngine* engine = runtime->deserializeCudaEngine(TRT_Model_Stream, size);
	assert(engine != nullptr);

	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);

	// ----- Free memory
	delete[] TRT_Model_Stream;
	TRT_Model_Stream = nullptr;

	auto out_dims = engine->getBindingDimensions(1);
	auto output_size = 1;
	for (int j = 0; j < out_dims.nbDims; j++)
	{
		output_size *= out_dims.d[j];
	}
	static float* output = new float[output_size];

	cv::VideoCapture cap(input_video_path);
	if (!cap.isOpened())
	{
		cout << "[E] open video failed!" << endl;
		return 0;
	}

	int img_w = (int)cap.get(CV_CAP_PROP_FRAME_WIDTH);
	int img_h = (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT);

	int fps = (int)cap.get(CV_CAP_PROP_FPS);
	cout << "FPS: " << fps << endl;
	long nFrame = static_cast<long>(cap.get(CV_CAP_PROP_FRAME_COUNT));
	cout << "Total frames: " << nFrame << endl;

	Mat img;
	int frame_id = 0;
	int total_ms = 0;

	// ---------- Define the tracker
	BYTETracker tracker(NUM_CLASSES, fps, TRACK_BUFFER,
		HIGH_DET_THRESH, NEW_TRACK_THRESH,
		HIGH_MATCH_THRESH, LOW_MATCH_THRESH,
		UNCONFIRMED_MATCH_THRESH);

	// Offline: using a video writer
	VideoWriter writer("./demo.mp4",
		CV_FOURCC('m', 'p', '4', 'v'),
		fps,
		Size(img_w, img_h));

	// ---------- Init mappings of CLASS_NAME and CLASS_ID
	initMappings();
	std::cout << "[I] mappings of class names and class ids are built.\n";

	while (true)
	{
		if (!cap.read(img))
		{
			break;
		}

		frame_id++;
		if (frame_id % 30 == 0)
		{
			std::cout << "Processing frame " << frame_id
				<< " (" << frame_id * 1000000 / total_ms << " fps)" << endl;
		}
		if (img.empty())
		{
			std::cout << "[Err]: empty image!" << endl;
			break;
		}

		// Resize image to fit net size
		Mat pr_img = staticResize(img);

		float* input;
		input = blobFromImage(pr_img);
		float scale = min(NET_W / float(img.cols),
			NET_H / float(img.rows));

		auto start = chrono::system_clock::now();

		// ---------- Run inference
		doInference(*context, output_size, pr_img.size(), input, output);
		// ----------

		// ---------- Decode outputs
		vector<Object> objects;
		decodeOutputs(output, scale, img_w, img_h, objects);

		if (TASK == TASK_MODES::TRACK)
		{
			// ---------- Update tracking results of current frame
			vector<Track> output_tracks;
			unordered_map<int, vector<Track>> output_tracks_dict;

			if (NUM_CLASSES == 1)  // Single class tracking output
			{
				output_tracks = tracker.update(objects);
			}
			else if (NUM_CLASSES > 1)  // Multi-class tracking output
			{
				output_tracks_dict = tracker.updateMCMOT(objects);
			}
			// ----------

			// update time
			auto end = chrono::system_clock::now();
			total_ms += (int)chrono::duration_cast<chrono::microseconds>(end - start).count();

			// ----- write tracking results for single class
			if (NUM_CLASSES == 1)
			{
				// Draw the tracking results
				drawTrackSC(output_tracks, frame_id, total_ms, img);

				if (VISUAL == VISUAL_MODES::OFFLINE)
				{
					writer.write(img);
				}
				else if (VISUAL == VISUAL_MODES::ONLINE)
				{
					if (frame_id == 1)
					{
						cv::namedWindow("Online", cv::WINDOW_AUTOSIZE);
					}
					cv::Mat img_rs;
					cv::resize(img, img_rs,
						cv::Size(int(img_w * RESIZE_RATIO), int(img_h * RESIZE_RATIO)),
						cv::INTER_AREA);
					cv::imshow("Online", img_rs);

					// Online play video or pause
					if (DELAY > 0 && waitKey(DELAY) == 32)
					{
						cv::waitKey(0);
					}
				}
			}
			else if (NUM_CLASSES > 1)  // multi-class tracking
			{
				// Draw the tracking results
				drawTrackMC(output_tracks_dict, frame_id, total_ms, img);

				if (VISUAL == VISUAL_MODES::OFFLINE)
				{
					writer.write(img);
				}
				else if (VISUAL == VISUAL_MODES::ONLINE)
				{
					if (frame_id == 1)
					{
						cv::namedWindow("Online", cv::WINDOW_AUTOSIZE);
					}
					cv::Mat img_rs;
					cv::resize(img, img_rs,
						cv::Size(int(img_w * RESIZE_RATIO), int(img_h * RESIZE_RATIO)),
						cv::INTER_AREA);
					cv::imshow("Online", img_rs);

					// Online play video or pause
					if (DELAY > 0 && waitKey(DELAY) == 32)
					{
						cv::waitKey(0);
					}
				}
			}
		}
		else if (TASK == TASK_MODES::DETECT)
		{
			// update time
			auto end = chrono::system_clock::now();
			total_ms += (int)chrono::duration_cast<chrono::microseconds>(end - start).count();

			// draw the detection results
			drawDetectResMC(objects, frame_id, total_ms, img);

			if (VISUAL == VISUAL_MODES::OFFLINE)
			{
				writer.write(img);
			}
			else if (VISUAL == VISUAL_MODES::ONLINE)
			{
				if (frame_id == 1)
				{
					cv::namedWindow("Online", cv::WINDOW_AUTOSIZE);
				}
				cv::Mat img_rs;
				cv::resize(img, img_rs,
					cv::Size(int(img_w * RESIZE_RATIO), int(img_h * RESIZE_RATIO)),
					cv::INTER_AREA);
				cv::imshow("Online", img_rs);

				// Online play video or pause
				if (DELAY > 0 && waitKey(DELAY) == 32)
				{
					cv::waitKey(0);
				}
			}
		}

		// free current frame's memory
		delete[] input;
		input = nullptr;

		waitKey(1);
		/*if (c > 0)
		{
			break;
		}*/
	}

	cap.release();
	cout << "FPS: " << frame_id * 1000000 / total_ms << endl;

	// destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();

	// free memory
	delete[] output;
	output = nullptr;

	return 0;
}


void decodeOutputs(const float* output_blob,
	const float& scale,
	const int& img_w,
	const int& img_h,
	vector<Object>& objects)
{
	vector<Object> proposals;
	const vector<int> strides({ 8, 16, 32 });
	vector<GridAndStride> grids_strides;
	generateGridsAndStride(NET_W, NET_H, strides, grids_strides);
	generateYoloxProposals(grids_strides,
		output_blob,
		(float)BBOX_CONF_THRESH,
		img_w, img_h,
		proposals);
	//std::cout << "num of boxes before nms: " << proposals.size() << std::endl;

	qsortDescentInplace(proposals);

	vector<int> picked;
	nmsSortedBboxes(proposals, picked, (float)NMS_THRESH);

	const int COUNT((int)picked.size());

	//std::cout << "num of boxes: " << count << std::endl;

	objects.resize(COUNT);
	for (int i = 0; i < COUNT; i++)
	{
		objects[i] = proposals[picked[i]];

		// adjust offset to original unpadded
		float x0 = (objects[i].rect.x) / scale;
		float y0 = (objects[i].rect.y) / scale;
		float x1 = (objects[i].rect.x + objects[i].rect.width) / scale;
		float y1 = (objects[i].rect.y + objects[i].rect.height) / scale;

		// clip
		// x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
		// y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
		// x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
		// y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

		objects[i].rect.x = x0;
		objects[i].rect.y = y0;
		objects[i].rect.width = x1 - x0;
		objects[i].rect.height = y1 - y0;
	}
}


void doInference(IExecutionContext& context,
	const int& output_size, const Size& input_shape,
	float* input, float* output)
{
	const ICudaEngine& engine = context.getEngine();

	// Pointers to input and output device buffers to pass to engine.
	// Engine requires exactly IEngine::getNbBindings() number of buffers.
	assert(engine.getNbBindings() == 2);
	void* buffers[2];

	// In order to bind the buffers, 
	// we need to know the names of the input and output tensors.
	// Note that indices are guaranteed to be less than IEngine::getNbBindings()
	const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
	assert(engine.getBindingDataType(inputIndex) == nvinfer1::DataType::kFLOAT);

	const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);
	assert(engine.getBindingDataType(outputIndex) == nvinfer1::DataType::kFLOAT);

	int mBatchSize = engine.getMaxBatchSize();

	// Create GPU buffers on device
	CHECK(cudaMalloc(&buffers[inputIndex],
		3 * input_shape.height * input_shape.width * sizeof(float)));
	CHECK(cudaMalloc(&buffers[outputIndex],
		output_size * sizeof(float)));

	// Create stream
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	// DMA input batch data to device, infer on the batch asynchronously,
	// and DMA output back to host
	CHECK(cudaMemcpyAsync(buffers[inputIndex],  // dst
		input,  // src
		3 * input_shape.height * input_shape.width * sizeof(float),
		cudaMemcpyHostToDevice,
		stream));

	// -----
	context.enqueue(1, buffers, stream, nullptr);
	// -----

	CHECK(cudaMemcpyAsync(output,  // dst
		buffers[outputIndex],  // src
		output_size * sizeof(float),
		cudaMemcpyDeviceToHost,
		stream));
	cudaStreamSynchronize(stream);

	// Release stream and buffers
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[inputIndex]));
	CHECK(cudaFree(buffers[outputIndex]));
}


void initMappings()
{
	for (int i = 0; i < CLASSES.size(); ++i)
	{
		CLASS2ID[CLASSES[i]] = i;
		ID2CLASS[i] = CLASSES[i];
	}
}


Mat staticResize(const Mat& img)
{
	float r = min((float)NET_W / float(img.cols), (float)NET_H / float(img.rows));
	// r = std::min(r, 1.0f);

	int unpad_w = int(r * img.cols);
	int unpad_h = int(r * img.rows);

	Mat re(unpad_h, unpad_w, CV_8UC3);
	cv::resize(img, re, re.size(), cv::INTER_AREA);

	Mat out(NET_H, NET_W, CV_8UC3, Scalar(114, 114, 114));
	re.copyTo(out(Rect(0, 0, re.cols, re.rows)));

	return out;
}


void generateGridsAndStride(const int& target_w, 
	const int& target_h,
	const vector<int>& strides,
	vector<GridAndStride>& grid_strides)
{
	for (auto stride : strides)
	{
		int num_grid_w = target_w / stride;
		int num_grid_h = target_h / stride;
		for (int g1 = 0; g1 < num_grid_h; g1++)
		{
			for (int g0 = 0; g0 < num_grid_w; g0++)
			{
				grid_strides.push_back(GridAndStride{ g0, g1, stride });
			}
		}
	}
}


void qsortDescentInplace(vector<Object>& faceobjects, int left, int right)
{
	int i = left;
	int j = right;
	float p = faceobjects[(left + right) / 2].prob;

	while (i <= j)
	{
		while (faceobjects[i].prob > p)
		{
			i++;
		}

		while (faceobjects[j].prob < p)
		{
			j--;
		}

		if (i <= j)
		{
			// swap
			swap(faceobjects[i], faceobjects[j]);
			i++;
			j--;
		}
	}

#pragma omp parallel sections
	{
#pragma omp section
		{
			if (left < j)
			{
				qsortDescentInplace(faceobjects, left, j);
			}
		}
#pragma omp section
		{
			if (i < right)
			{
				qsortDescentInplace(faceobjects, i, right);
			}
		}
	}
}


void qsortDescentInplace(vector<Object>& objects)
{
	if (objects.empty())
	{
		return;
	}
	qsortDescentInplace(objects, 0, (int)objects.size() - 1);
}


void nmsSortedBboxes(const vector<Object>& faceobjects,
	vector<int>& picked,
	float nms_threshold)
{
	picked.clear();

	const int n = (int)faceobjects.size();

	vector<float> areas(n);
	for (int i = 0; i < n; i++)
	{
		areas[i] = faceobjects[i].rect.area();
	}

	for (int i = 0; i < n; i++)
	{
		const Object& a = faceobjects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++)
		{
			const Object& b = faceobjects[picked[j]];

			// intersection over union
			float inter_area = intersectionArea(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
			{
				keep = 0;
			}
		}

		if (keep)
		{
			picked.push_back(i);
		}
	}
}


void generateYoloxProposals(const vector<GridAndStride>& grids_strides,
	const float* output_blob,
	const float& prob_threshold,
	const int& img_w, const int& img_h,
	vector<Object>& objects)
{
	// ----- Traverse all anchors of 3 scales(/8, /16, /32)
	const int N_ANCHORS = (int)grids_strides.size();
	for (int anchor_idx = 0; anchor_idx < N_ANCHORS; anchor_idx++)
	{
		const int& grid_x = grids_strides[anchor_idx].grid_x;
		const int& grid_y = grids_strides[anchor_idx].grid_y;
		const int& stride = grids_strides[anchor_idx].stride;
		const int basic_pos = anchor_idx * (NUM_CLASSES + 5);

		// ----- yolox/models/yolo_head.py decode logic
		float x_center = (output_blob[basic_pos + 0] + grid_x) * stride;
		float y_center = (output_blob[basic_pos + 1] + grid_y) * stride;
		float w = exp(output_blob[basic_pos + 2]) * stride;
		float h = exp(output_blob[basic_pos + 3]) * stride;

		if (x_center <= 0 || y_center <= 0 
			|| x_center >= img_w || y_center >= img_h
			|| w <= stride || h <= stride 
			|| w >= img_w || h >= img_h)
		{
			continue;
		}

		float x0 = x_center - w * 0.5f;
		float y0 = y_center - h * 0.5f;

		const float& box_objectness = output_blob[basic_pos + 4];
		if (box_objectness <= 0.0f)
		{
			continue;
		}
		for (int class_idx = 0; class_idx < NUM_CLASSES; class_idx++)
		{
			const float& box_cls_score = output_blob[basic_pos + 5 + class_idx];
			if (box_cls_score <= 0.0f)
			{
				continue;
			}
			float box_prob = box_objectness * box_cls_score;
			if (box_prob > 1.0f || isnan(box_prob))
			{
				continue;
			}
			if (box_prob > prob_threshold)
			{
				// Fill the object fields and save
				Object obj;
				obj.label = class_idx;  // class index
				obj.rect.x = x0;
				obj.rect.y = y0;
				obj.rect.width = w;
				obj.rect.height = h;
				obj.label = class_idx;
				obj.prob = box_prob;
				objects.push_back(obj);
			}

		} // class loop

	} // point anchor loop
}


float* blobFromImage(Mat& img)
{
	// Get RGB image data
	cvtColor(img, img, COLOR_BGR2RGB);

	float* blob = new float[img.total() * 3];
	int channels = 3;
	const int& img_h = img.rows;
	const int& img_w = img.cols;

	vector<float> mean({ 0.485f, 0.456f, 0.406f });
	vector<float> std({ 0.229f, 0.224f, 0.225f });

	for (int c = 0; c < channels; c++)
	{
		for (int h = 0; h < img_h; h++)
		{
			for (int w = 0; w < img_w; w++)
			{
				blob[c * img_w * img_h + h * img_w + w] =
					(((float)img.at<Vec3b>(h, w)[c]) / 255.0f - mean[c]) / std[c];
			}
		}
	}

	return blob;
}


void drawTrackSC(const std::vector<Track>& output_tracks,
	const int& num_frames, const int& total_ms,
	cv::Mat& img)
{
	for (int i = 0; i < output_tracks.size(); ++i)
	{
		Scalar s = BYTETracker::getColor(output_tracks[i].track_id);
		const vector<float>& tlwh = output_tracks[i].tlwh;

		// Draw class name
		cv::putText(img,
			ID2CLASS[output_tracks[i].class_id],
			Point((int)tlwh[0], (int)tlwh[1] - 5),
			0,
			0.6,
			Scalar(0, 255, 255),
			2,
			cv::LINE_AA);

		// Draw track id
		cv::putText(img,
			format("%d", output_tracks[i].track_id),  // track id
			Point((int)tlwh[0], (int)tlwh[1] - 12),
			0,
			0.6,
			Scalar(0, 255, 255),
			2,
			cv::LINE_AA);

		// Draw bounding box
		cv::rectangle(img,
			cv::Rect((int)tlwh[0], (int)tlwh[1], (int)tlwh[2], (int)tlwh[3]),
			s,
			2);

	}

	cv::putText(img,
		format("frame: %d fps: %d num: %d",
			num_frames, num_frames * 1000000 / total_ms, output_tracks.size()),
		Point(0, 30),
		0,
		0.6,
		Scalar(0, 0, 255),
		2,
		LINE_AA);
}


void drawTrackMC(const std::unordered_map<int, vector<Track>>& output_tracks_dict,
	const int& num_frames, const int& total_ms,
	cv::Mat& img)
{
	int total_obj_count = 0;

	// hash table traversing
	for (auto it = output_tracks_dict.begin(); it != output_tracks_dict.end(); it++)
	{
		const vector<Track>& output_tracks = it->second;
		total_obj_count += (int)output_tracks.size();
		for (int i = 0; i < output_tracks.size(); ++i)
		{
			Scalar s = BYTETracker::getColor(output_tracks[i].track_id);
			const vector<float>& tlwh = output_tracks[i].tlwh;
			//const int& x0 = tlwh[0];

			// Draw class name
			cv::putText(img,
				ID2CLASS[output_tracks[i].class_id],
				Point((int)tlwh[0], (int)tlwh[1] - 5),
				0,
				0.6,
				Scalar(0, 255, 255),
				2,
				cv::LINE_AA);

			// Draw track id
			cv::putText(img,
				format("%d", output_tracks[i].track_id),  // track id
				Point((int)tlwh[0], (int)tlwh[1] - 15),
				0,
				0.6,
				Scalar(0, 255, 255),
				2,
				cv::LINE_AA);

			// Draw bounding box
			cv::rectangle(img,
				cv::Rect((int)tlwh[0], (int)tlwh[1], (int)tlwh[2], (int)tlwh[3]),
				s,
				2);
			
		}
	}

	cv::putText(img,
		format("frame: %d fps: %d num: %d",
			num_frames, num_frames * 1000000 / total_ms, total_obj_count),
		Point(0, 30),
		0,
		0.6,
		Scalar(0, 0, 255),
		2,
		LINE_AA);
}

void drawDetectResMC(const vector<Object>& objects,
	const int& num_frames,
	const int& total_ms, 
	cv::Mat& img)
{
	for (int i = 0; i < objects.size(); ++i)
	{
		const Object& obj = objects[i];
		const cv::Rect& rect = obj.rect;

		Scalar s = BYTETracker::getColor(i);

		// Draw class name
		cv::putText(img,
			ID2CLASS[obj.label],
			Point(rect.x, rect.y - 5),
			0,
			0.6,
			Scalar(0, 255, 255),
			2,
			cv::LINE_AA);

		char prob_chars[50];
		sprintf(prob_chars, "%.3f", obj.prob);
		cv::putText(img,
			prob_chars,
			Point(rect.x, rect.y - 20),
			0,
			0.6,
			Scalar(0, 255, 255),
			2,
			cv::LINE_AA);

		// Draw bounding box
		cv::rectangle(img,
			rect,
			s,
			2);
	}

	cv::putText(img,
		format("frame: %d fps: %d num: %d",
			num_frames, num_frames * 1000000 / total_ms, objects.size()),
		Point(0, 30),
		0,
		0.6,
		Scalar(0, 0, 255),
		2,
		LINE_AA);
}


//if (argc == 4 && string(argv[2]) == "-i")
//{
//	const string engine_file_path{ argv[1] };
//	ifstream file(engine_file_path, ios::binary);
//	if (file.good())
//	{
//		file.seekg(0, file.end);
//		size = file.tellg();
//		file.seekg(0, file.beg);
//		trtModelStream = new char[size];
//		assert(trtModelStream);
//		file.read(trtModelStream, size);
//		file.close();
//		cout << "[I] " << engine_file_path << " loaded." << endl;
//	}
//	else
//	{
//		cout << "[Err]: read engine file " << engine_file_path << " failed!" << endl;
//	}
//}
//else
//{
//	cerr << "arguments not right!" << endl;
//	cerr << "run 'python3 tools/trt.py -f exps/example/mot/yolox_s_mix_det.py -c pretrained/bytetrack_s_mot17.pth.tar' to serialize model first!" << std::endl;
//	cerr << "Then use the following command:" << endl;
//	cerr << "cd demo/TensorRT/cpp/build" << endl;
//	cerr << "./bytetrack ../../../../YOLOX_outputs/yolox_s_mix_det/model_trt.engine -i ../../../../videos/palace.mp4  // deserialize file and run inference" << std::endl;
//	return -1;
//}
//const string input_video_path{ argv[3] };  // video path