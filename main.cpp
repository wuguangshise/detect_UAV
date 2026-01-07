#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class YOLO11Detector
{
public:
    YOLO11Detector(string modelPath, string labelPath, float confThreshold = 0.25f, float nmsThreshold = 0.45f);
    void detect(Mat& frame);

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


void YOLO11Detector::detect(Mat& frame)
{
    if (frame.empty()) return;

    Mat blob = blobFromImage(frame, 1 / 255.0, Size(this->inpWidth, this->inpHeight), Scalar(0, 0, 0), true, false);
    this->net.setInput(blob);


    vector<Mat> outs;
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());

    if (outs.empty()) return;

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
    }
}

int main()
{
	// 配置路径,按照使用的硬件的路径修改
    string model = "D:\\yolov13\\yolov13-main\\YOLO_UAV_Distillation\\distill_x_to_n_fixed\\weights\\best.onnx";
    string labels = "C:/Users/Administrator/Desktop/c++UAV/label.txt";
    string imgPath = "D:/dataset/UAV/yolo_format/test/images/00277.jpg";

    try {
        YOLO11Detector detector(model, labels, 0.3f, 0.45f);

        Mat frame = imread(imgPath);
        if (frame.empty()) {
            cerr << "无法读取测试图片: " << imgPath << endl;
            return -1;
        }

        detector.detect(frame);

        namedWindow("Object Detection", WINDOW_NORMAL);
        imshow("Object Detection", frame);

        std::cout << "检测完成，按任意键退出。" << std::endl;
        cv::waitKey(0);
    }
    catch (const exception& e) {
        cerr << "程序发生错误: " << e.what() << endl;
        return -1;
    }

    return 0;
}