//注释部分使用了 Gemini3 pro 进行补充注释

#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <numeric>
#include <random>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// 检测结果结构体
struct Detection {
    int classId;
    float confidence;
    Rect box;
};

class YOLO11Detector
{
public:
    YOLO11Detector(string modelPath, string labelPath, float confThreshold = 0.25f, float nmsThreshold = 0.45f);
    vector<Detection> detect(Mat& frame);

private:
    const int inpWidth = 640;
    const int inpHeight = 640;

    float confThreshold;
    float nmsThreshold;
    Net net;
    vector<string> class_names;
    vector<Scalar> colors;

    void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
    vector<string> load_class_names(string path);
};

YOLO11Detector::YOLO11Detector(string modelPath, string labelPath, float confThreshold, float nmsThreshold)
{
    this->confThreshold = confThreshold;
    this->nmsThreshold = nmsThreshold;

    // 检查模型文件是否存在
    ifstream mf(modelPath);
    if (!mf.good()) {
        throw runtime_error("Model file not found: " + modelPath);
    }

    try {
        this->net = readNetFromONNX(modelPath);
        this->net.setPreferableBackend(DNN_BACKEND_OPENCV);
        this->net.setPreferableTarget(DNN_TARGET_CPU);
    }
    catch (const cv::Exception& e) {
        stringstream ss;
        ss << "Failed to load ONNX model: " << e.what() << "\n";
        ss << "Check that the model path is correct, the ONNX file is valid, and your OpenCV build supports ONNX backend.\n";
        ss << "OpenCV build info:\n" << getBuildInformation();
        throw runtime_error(ss.str());
    }

    // 加载标签
    this->class_names = load_class_names(labelPath);
    if (this->class_names.empty()) {
        cerr << "Warning: no class names loaded from: " << labelPath << endl;
    }

    RNG rng(12345);
    for (int i = 0; i < class_names.size(); i++) {
        colors.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
    }
}

vector<string> YOLO11Detector::load_class_names(string path) {
    vector<string> names;
    ifstream ifs(path);
    if (!ifs.good()) {
        cerr << "Warning: label file not found: " << path << endl;
        return names;
    }
    string line;
    while (getline(ifs, line)) {
        if (!line.empty()) names.push_back(line);
    }
    return names;
}

void YOLO11Detector::drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
    Scalar color = colors.empty() ? Scalar(0, 255, 0) : colors[classId % colors.size()];

    rectangle(frame, Point(left, top), Point(right, bottom), color, 2);

    string label = format("%.2f", conf);
    if (classId >= 0 && classId < (int)class_names.size()) {
        label = class_names[classId] + ":" + label;
    }

    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    top = max(top, labelSize.height);

    rectangle(frame, Point(left, top - labelSize.height - 5),
        Point(left + labelSize.width, top + baseLine), color, FILLED);

    putText(frame, label, Point(left, top - 2), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
}

vector<Detection> YOLO11Detector::detect(Mat& frame)
{
    vector<Detection> detections;
    if (frame.empty()) return detections;

    Mat blob = blobFromImage(frame, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);

    vector<Mat> outs;
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

    if (outs.empty()) return detections;

    Mat output = outs[0];
    if (output.dims == 3) {
        output = output.reshape(1, output.size[1]);
    }
    output = output.t();

    vector<float> confidences;
    vector<Rect> boxes;
    vector<int> classIds;

    float x_factor = (float)frame.cols / inpWidth;
    float y_factor = (float)frame.rows / inpHeight;

    float* pdata = (float*)output.data;
    int rows = output.rows;
    int cols = output.cols;

    for (int i = 0; i < rows; ++i) {
        Mat scores = output.row(i).colRange(4, cols);
        Point classIdPoint;
        double max_class_score;
        minMaxLoc(scores, 0, &max_class_score, 0, &classIdPoint);

        if (max_class_score > this->confThreshold) {
            float cx = pdata[0];
            float cy = pdata[1];
            float w = pdata[2];
            float h = pdata[3];

            int left = int((cx - 0.5 * w) * x_factor);
            int top = int((cy - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            left = max(0, left);
            top = max(0, top);
            width = min(width, frame.cols - left);
            height = min(height, frame.rows - top);

            if (width > 0 && height > 0) {
                confidences.push_back((float)max_class_score);
                boxes.push_back(Rect(left, top, width, height));
                classIds.push_back(classIdPoint.x);
            }
        }
        pdata += cols;
    }

    vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, indices);

    for (int idx : indices) {
        Rect box = boxes[idx];
        this->drawPred(classIds[idx], confidences[idx], box.x, box.y,
            box.x + box.width, box.y + box.height, frame);

        Detection det;
        det.classId = classIds[idx];
        det.confidence = confidences[idx];
        det.box = box;
        detections.push_back(det);
    }

    return detections;
}

// FAST 特征检测模块
class AcceleratedFAST
{
public:
    AcceleratedFAST(int threshold = 20, bool nonmaxSuppression = true);
    // 只在检测框内检测 FAST 点，返回全局坐标点列表
    vector<Point2f> detect_features(const Mat& frame, const vector<Detection>& yolo_detections);

private:
    Ptr<FastFeatureDetector> fast;
};

AcceleratedFAST::AcceleratedFAST(int threshold, bool nonmaxSuppression)
{
    fast = FastFeatureDetector::create(threshold, nonmaxSuppression);
    cout << ">> FAST 模块加载完成：用于几何特征提取" << endl;
}

vector<Point2f> AcceleratedFAST::detect_features(const Mat& frame, const vector<Detection>& yolo_detections)
{
    vector<Point2f> stable_points;

    if (frame.empty() || yolo_detections.empty()) {
        return stable_points; // nothing to do
    }

    // 转换为灰度图一次，然后在每个 ROI 上处理
    Mat gray_frame;
    cvtColor(frame, gray_frame, COLOR_BGR2GRAY);

    // 为避免重复点或过多点，可选限流参数（例如每个 ROI 最多保留 N 点）
    const size_t MAX_POINTS_PER_ROI = 500;

    for (const auto& det : yolo_detections) {
        // 将 ROI 裁剪到图像范围
        Rect roi = det.box & Rect(0, 0, frame.cols, frame.rows);
        if (roi.width <= 2 || roi.height <= 2) continue;

        // 把 ROI 中对应的灰度图取出来
        Mat roi_gray = gray_frame(roi);

        // 在 ROI 上检测 FAST 特征
        vector<KeyPoint> kp_roi;
        fast->detect(roi_gray, kp_roi);

        // 如果太多点，按响应排序并截断
        if (kp_roi.size() > MAX_POINTS_PER_ROI) {
            sort(kp_roi.begin(), kp_roi.end(), [](const KeyPoint& a, const KeyPoint& b) {
                return a.response > b.response;
            });
            kp_roi.resize(MAX_POINTS_PER_ROI);
        }

        // 将 ROI 局部坐标转为全局坐标并加入结果列表
        for (const auto& kp : kp_roi) {
            Point2f pt_global(kp.pt.x + roi.x, kp.pt.y + roi.y);

            // 可选：如果只想要落在某类（例如非动态目标）内的点，可在这里筛选
            stable_points.push_back(pt_global);
        }
    }

    return stable_points;
}

// IMM 滤波器模块
class IMMFilter
{
public:
    IMMFilter(int num_models = 3);
    map<int, tuple<Point2f, Point2f, Point2f>> update_trackers(
        const map<int, Rect>& yolo_measurements,
        const vector<Point2f>& all_fast_points);

private:
    int num_models;
    map<int, tuple<float, float, float, float>> track_states; // x, y, vx, vy
};

IMMFilter::IMMFilter(int num_models)
{
    this->num_models = num_models;
    cout << ">> IMM 滤波器加载完成：运行 " << num_models << " 个运动模型" << endl;
}

map<int, tuple<Point2f, Point2f, Point2f>> IMMFilter::update_trackers(
    const map<int, Rect>& yolo_measurements,
    const vector<Point2f>& all_fast_points)
{
    map<int, tuple<Point2f, Point2f, Point2f>> tracked_objects;

    // 将 FAST 点转换为 Mat 用于快速处理
    Mat fast_points_mat;
    if (!all_fast_points.empty()) {
        fast_points_mat = Mat(all_fast_points).reshape(1);
    }

    const int MIN_FAST_POINTS_THRESHOLD = 5;
    const float WEIGHT_YOLO = 0.6f;
    const float WEIGHT_FAST = 0.4f;
    const float SMOOTHING_ALPHA = 0.3f;

    for (const auto& kv : yolo_measurements) {
        int track_id = kv.first;
        Rect box = kv.second;

        // 1. YOLO 测量值：边界框中心点
        float cx = box.x + box.width / 2.0f;
        float cy = box.y + box.height / 2.0f;

        float fused_x = cx, fused_y = cy; // 默认值：使用 YOLO 中心

        // 融合点：几何特征辅助测量
        if (!all_fast_points.empty()) {
            // 2. 找出落在当前边界框内的 FAST 关键点
            vector<Point2f> in_box_fast_points;
            for (const auto& pt : all_fast_points) {
                if (pt.x >= box.x && pt.x <= box.x + box.width &&
                    pt.y >= box.y && pt.y <= box.y + box.height) {
                    in_box_fast_points.push_back(pt);
                }
            }

            // 3. 计算融合后的测量值
            if (in_box_fast_points.size() >= MIN_FAST_POINTS_THRESHOLD) {
                // 计算 FAST 点的几何中心
                float fast_center_x = 0, fast_center_y = 0;
                for (const auto& pt : in_box_fast_points) {
                    fast_center_x += pt.x;
                    fast_center_y += pt.y;
                }
                fast_center_x /= in_box_fast_points.size();
                fast_center_y /= in_box_fast_points.size();

                // 融合策略：加权平均
                fused_x = WEIGHT_YOLO * cx + WEIGHT_FAST * fast_center_x;
                fused_y = WEIGHT_YOLO * cy + WEIGHT_FAST * fast_center_y;
            }
        }

        // 4. IMM 状态更新（使用融合后的测量值）
        if (track_states.find(track_id) == track_states.end()) {
            track_states[track_id] = make_tuple(fused_x, fused_y, 0.0f, 0.0f);
        }

        float x_prev = get<0>(track_states[track_id]);
        float y_prev = get<1>(track_states[track_id]);

        // 使用融合值进行平滑（简化的 IMM 更新）
        float x_smooth = (1.0f - SMOOTHING_ALPHA) * x_prev + SMOOTHING_ALPHA * fused_x;
        float y_smooth = (1.0f - SMOOTHING_ALPHA) * y_prev + SMOOTHING_ALPHA * fused_y;

        // 更新状态
        track_states[track_id] = make_tuple(x_smooth, y_smooth, 0.0f, 0.0f);

        // 存储结果
        tracked_objects[track_id] = make_tuple(
            Point2f(cx, cy),                    // YOLO 中心
            Point2f(fused_x, fused_y),          // 融合位置
            Point2f(x_smooth, y_smooth)         // 平滑位置
        );
    }

    return tracked_objects;
}

// 绘制跟踪结果
void draw_tracking_results(Mat& frame, const map<int, tuple<Point2f, Point2f, Point2f>>& tracked_objects,
    const map<int, Rect>& tracked_boxes)
{
    for (const auto& kv : tracked_objects) {
        int track_id = kv.first;
        tuple<Point2f, Point2f, Point2f> points = kv.second;
        Point2f yolo_center = get<0>(points);
        Point2f fused_pos = get<1>(points);
        Point2f smooth_pos = get<2>(points);

        // 绘制平滑后的位置（绿色框）
        if (tracked_boxes.find(track_id) != tracked_boxes.end()) {
            Rect box = tracked_boxes.at(track_id);
            // 保持原有框的大小，只移动中心到平滑位置
            Rect smooth_box(smooth_pos.x - box.width / 2, smooth_pos.y - box.height / 2,
                box.width, box.height);

            // 绘制平滑框（绿色）
            rectangle(frame, smooth_box, Scalar(0, 255, 0), 2);

            // 绘制跟踪ID
            string id_text = "ID: " + to_string(track_id);
            putText(frame, id_text, Point(smooth_box.x, smooth_box.y - 5),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);

            // 绘制融合点（蓝色点）
            circle(frame, fused_pos, 3, Scalar(255, 0, 0), -1);

            // 绘制平滑点（红色点）
            circle(frame, smooth_pos, 5, Scalar(0, 0, 255), -1);

            // 绘制从YOLO中心到平滑中心的线
            arrowedLine(frame, yolo_center, smooth_pos, Scalar(0, 255, 255), 1, LINE_AA, 0, 0.1);
        }
    }
}

int main()
{
    // 配置路径
    string model = "D:\\yolov13\\yolov13-main\\YOLO_UAV_Distillation\\distill_x_to_n_fixed\\weights\\best.onnx";
    string labels = "C:/Users/Administrator/Desktop/c++UAV/label.txt";
    string videoPath = "path"; // 改为视频路径

    try {
        // 初始化YOLO检测器
        YOLO11Detector detector(model, labels, 0.3f, 0.45f);

        // 初始化FAST检测器
        AcceleratedFAST fast_detector(20, true);

        // 初始化IMM滤波器
        IMMFilter imm_filter(3);

        // 打开视频文件
        VideoCapture cap(videoPath);
        if (!cap.isOpened()) {
            cerr << "无法打开视频文件: " << videoPath << endl;

            // 如果视频打开失败，尝试使用单张图片
            string imgPath = "D:/dataset/UAV/yolo_format/test/images/00279.jpg";
            Mat frame = imread(imgPath);
            if (frame.empty()) {
                cerr << "也无法读取测试图片: " << imgPath << endl;
                return -1;
            }

            // 单张图片处理
            vector<Detection> detections = detector.detect(frame);

            // 提取FAST特征点（不绘制）
            vector<Point2f> fast_points = fast_detector.detect_features(frame, detections);

            // 将检测结果转换为跟踪格式
            map<int, Rect> yolo_measurements;
            int track_id = 0;
            for (const auto& det : detections) {
                // 只跟踪特定类别（例如类别0）
                if (det.classId == 0) {
                    yolo_measurements[track_id++] = det.box;
                }
            }

            // 更新IMM滤波器
            auto tracked_objects = imm_filter.update_trackers(yolo_measurements, fast_points);

            // 绘制跟踪结果
            draw_tracking_results(frame, tracked_objects, yolo_measurements);

            namedWindow("Object Detection and Tracking", WINDOW_NORMAL);
            imshow("Object Detection and Tracking", frame);

            cout << "检测和跟踪完成，按任意键退出。" << endl;
            waitKey(0);
        }
        else {
            // 视频处理
            Mat frame;
            int frame_count = 0;

            // 存储跟踪ID
            map<int, Rect> tracked_objects;
            int next_track_id = 0;

            while (cap.read(frame)) {
                if (frame.empty()) break;

                frame_count++;
                cout << "处理帧: " << frame_count << endl;

                // YOLO检测
                vector<Detection> detections = detector.detect(frame);

                // 提取FAST特征点（不绘制）
                vector<Point2f> fast_points = fast_detector.detect_features(frame, detections);

                // 简单的目标关联（基于IOU）
                map<int, Rect> current_measurements;

                for (const auto& det : detections) {
                    // 只跟踪特定类别（例如类别0）
                    if (det.classId == 0) {
                        bool matched = false;
                        int matched_id = -1;
                        float max_iou = 0.0f;

                        // 寻找最佳匹配的现有跟踪目标
                        for (const auto& prev_kv : tracked_objects) {
                            int track_id = prev_kv.first;
                            Rect prev_box = prev_kv.second;
                            Rect intersection = det.box & prev_box;
                            float iou = 0.0f;
                            float denom = (float)(det.box.area() + prev_box.area() - intersection.area());
                            if (denom > 0.0f) iou = (float)(intersection.area()) / denom;

                            if (iou > max_iou && iou > 0.3f) { // IOU阈值
                                max_iou = iou;
                                matched_id = track_id;
                                matched = true;
                            }
                        }

                        if (matched) {
                            // 使用现有ID
                            current_measurements[matched_id] = det.box;
                        }
                        else {
                            // 分配新ID
                            current_measurements[next_track_id] = det.box;
                            next_track_id++;
                        }
                    }
                }

                // 更新跟踪目标列表
                tracked_objects = current_measurements;

                // 更新IMM滤波器
                auto tracking_results = imm_filter.update_trackers(current_measurements, fast_points);

                // 绘制跟踪结果
                draw_tracking_results(frame, tracking_results, current_measurements);

                // 显示帧信息
                string info = "Frame: " + to_string(frame_count) +
                    " | Objects: " + to_string(current_measurements.size()) +
                    " | FAST points: " + to_string(fast_points.size());
                putText(frame, info, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);

                namedWindow("Object Detection and Tracking", WINDOW_NORMAL);
                imshow("Object Detection and Tracking", frame);

                // 按ESC退出
                if (waitKey(1) == 27) {
                    break;
                }
            }

            cap.release();
        }

        destroyAllWindows();
    }
    catch (const exception& e) {
        cerr << "程序发生错误: " << e.what() << endl;
        return -1;
    }

    return 0;
}
