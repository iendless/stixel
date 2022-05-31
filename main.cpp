#include <iostream>
#include <fstream>
#include <chrono>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "semi_global_matching.h"
#include "stixel_world.h"

#include <dirent.h>
#include <assert.h>
using namespace cv;
using namespace std;


//* 最外层, 输入输出实验数据有关的参数放在全局
string directory;
string mode;
string dispDir;
string stixelDir;
DIR *dp, *dp1, *dp2;     // * 文件夹指针
struct dirent *ep;  // * 用于获取文件名
string image2_dir;
string image3_dir;
string tiff_dir;
ofstream bboxFile;
ofstream dispFile;
SemiGlobalMatching::Parameters param;




void usage(char* argv[])
{
    //Folder struct
    //-directory/
    //     --image2/
    //     --image3/
    std::cout << "usage: " << argv[0] << "stixel [directory] [camera in/ex para]" << std::endl;

    cout<<"Foler structure"<<endl;
    cout<<"-[directory]/"<<endl;
    cout<<"      --image_2/"<<endl;
    cout<<"      --image_3/"<<endl;
}

static cv::Scalar computeColor(float val)
{
	const float hscale = 6.f;
	float h = 0.6f * (1.f - val), s = 1.f, v = 1.f;
	float r, g, b;

	static const int sector_data[][3] =
	{ { 1,3,0 },{ 1,0,2 },{ 3,0,1 },{ 0,2,1 },{ 0,1,3 },{ 2,1,0 } };
	float tab[4];
	int sector;
	h *= hscale;
	if (h < 0)
		do h += 6; while (h < 0);
	else if (h >= 6)
		do h -= 6; while (h >= 6);
	sector = cvFloor(h);
	h -= sector;
	if ((unsigned)sector >= 6u)
	{
		sector = 0;
		h = 0.f;
	}

	tab[0] = v;
	tab[1] = v*(1.f - s);
	tab[2] = v*(1.f - s*h);
	tab[3] = v*(1.f - s*(1.f - h));

	b = tab[sector_data[sector][0]];
	g = tab[sector_data[sector][1]];
	r = tab[sector_data[sector][2]];
	return 255 * cv::Scalar(b, g, r);
}

static cv::Scalar dispToColor(float disp, float maxdisp)
{
	if (disp < 0)
		return cv::Scalar(128, 128, 128);
	return computeColor(std::min(disp, maxdisp) / maxdisp);
}

static void drawStixel(cv::Mat& img, const Stixel& stixel, cv::Scalar color)
{
	const int radius = std::max(stixel.width / 2, 1);
	const cv::Point tl(stixel.u - radius, stixel.vT);
	const cv::Point br(stixel.u + radius, stixel.vB);
	cv::rectangle(img, cv::Rect(tl, br), color, -1);
}

static void init_stix_param(StixelWorld::Parameters& stix_param, const cv::FileStorage& cvfs) {
    stix_param.camera.fu = cvfs["FocalLengthX"];
    stix_param.camera.fv = cvfs["FocalLengthY"];
    stix_param.camera.u0 = cvfs["CenterX"];
    stix_param.camera.v0 = cvfs["CenterY"];
    stix_param.camera.baseline = cvfs["BaseLine"];
    stix_param.camera.height = cvfs["Height"];
    stix_param.camera.tilt = cvfs["Tilt"];
    stix_param.minDisparity = -1;
    stix_param.maxDisparity = 64;
}

static cv::Mat depthToDisp(cv::Mat& depthMat, cv::Mat dispMat, CoordinateTransform coordinateTransform) {
    cout << "depth channels: " << depthMat.channels() << endl;
    int height = depthMat.rows;
    int width = depthMat.cols;

    float fu = coordinateTransform.camera.fu;
    float baseline = coordinateTransform.camera.baseline;

    for(int i = 0; i < height; i++) {
        for(int j = 0 ; j < width; j++) {
            float curDep = depthMat.at<float>(i, j);
            if(!isnan(curDep) && !isinf(curDep)) {
                dispMat.at<float>(i, j) = coordinateTransform.toD(0, curDep);
                continue;
                // cout << dispMat.at<float>(i, j) << endl;
            }
            if(isnan(curDep)) {
                // cout << "bad";
                // cout << curDep << endl;
                dispMat.at<float>(i, j) = -1;
                continue;
            }
            if(isinf(curDep)) {
                dispMat.at<float>(i, j) = 0;
                continue;
            }
        }
    }
    return dispMat;
}

static void outputImg(Mat& img, string dir, string dispName, int mode) {
    string filePath = dir + dispName;
    imwrite(filePath, img);
    string dispDataName(dispName.begin(), dispName.end()-4);
    dispDataName += ".txt";
    // * 输出disp 数据值到txt 的代码，暂时不用
    if(mode == 1) {
        ofstream dispFile(dir+dispDataName, ios::app);
        dispFile << cv::format(img, cv::Formatter::FMT_DEFAULT) << endl;
        dispFile.flush();
    }
}

static void outputDisp(Mat& D0, Mat& draw, string image_name) {


    // 视差输出
    D0.convertTo(draw, CV_8U, 255. / (SemiGlobalMatching::DISP_SCALE * param.numDisparities));
    cv::applyColorMap(draw, draw, cv::COLORMAP_JET);   
    draw.setTo(0, D0 == SemiGlobalMatching::DISP_INV);
    Mat fdisp;
    D0.convertTo(fdisp, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);
    string dispName(image_name.begin(), image_name.end()-4);
    dispFile = ofstream(dispDir+dispName+".txt", ios::app);
    dispFile << cv::format(fdisp, cv::Formatter::FMT_DEFAULT) << endl;
    dispName += "Disp.png";
    outputImg(fdisp, stixelDir, dispName);
    
    cout << "D0 channels " << D0.channels() << endl;
    cout << "fdisp channels " << fdisp.channels() << endl;
}

static Mat sgmComputeDisp(string image_name, Mat& D0, Mat& D1) {
    if (!strcmp (ep->d_name, "."))
            exit(0);
    if (!strcmp (ep->d_name, ".."))
        exit(0);

    string I0_path = directory + "/" + "image_2" + "/" + image_name;
    string I1_path = directory + "/" + "image_3" + "/" + image_name;
    SemiGlobalMatching::Parameters param;
    SemiGlobalMatching sgm(param);
    bboxFile << image_name << endl;
    cout << image_name.c_str() << endl;
    cout<<"I0: "<<I0_path<<endl;
    cout<<"I1: "<<I1_path<<endl;

    Mat I0 = imread(I0_path);   
    Mat I1 = imread(I1_path);

    if (I0.empty() || I1.empty())
    {
        std::cerr << "failed to read any image." << std::endl;
        exit(0);
    }
    CV_Assert(I0.size() == I1.size() && I0.type() == I1.type());

    // 视差计算
    Mat I0_Gray, I1_Gray;
    cvtColor(I0, I0_Gray, cv::COLOR_BGR2GRAY);
    cvtColor(I1, I1_Gray, cv::COLOR_BGR2GRAY);
    
    sgm.compute(I0_Gray, I1_Gray, D0, D1);    // ! 双目图像匹配 SGM算法
    return I0;
}

static Mat depthComputeDisp(string image_name, Mat& D0, CoordinateTransform& coordinateTransform) {
    cv::Mat depthTiff = imread(directory+tiff_dir+image_name, IMREAD_UNCHANGED);
    string I0_path = directory + "/" + "image_2" + "/" + image_name;
    Mat I0 = imread(I0_path);
    D0 = depthTiff.clone();
    depthToDisp(depthTiff, D0, coordinateTransform);
    return I0;
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        usage(argv);
        return -1;
    }

    // string directory = "/home/endless/stixel/data/";
    directory = argv[1];
    mode = argv[2];
    if(argv[2] == "smg") {
        dispDir = directory + "smgDisp/";
        stixelDir = directory + "smgStixel/";
    } else if(argv[2] == "tiff") {
        dispDir = directory + "tiffDisp/";
        stixelDir = directory + "tiffStixel/";
    } else {
        cout << "wrong arg, comfirm them" << endl;
    }

    image2_dir = directory + "/" + "image_2";
    dp = opendir(image2_dir.c_str());

    image3_dir = directory + "/" + "image_3";
    dp1 = opendir(image3_dir.c_str());

    tiff_dir = directory + "/" + "tiff";
    dp2 = opendir(tiff_dir.c_str());

    if (dp == NULL || dp1 ==  NULL || dp2 == NULL) {
        std::cerr << "Invalid folder structure under: " << directory << std::endl;
        usage(argv);
        exit(EXIT_FAILURE); 
    }

    cv::Mat D0, D1, draw, depthDiff;
    bboxFile = ofstream(directory + "stixelBBox.txt", ios::app);
    const cv::FileStorage cvfs(argv[2], FileStorage::READ);
    // !
    StixelWorld::Parameters stix_param;
    assert(cvfs.isOpened() == true);
    init_stix_param(stix_param, cvfs);
    CoordinateTransform coordinateTransform(stix_param.camera);

    
    while ((ep = readdir(dp)) != NULL) 
    {
        // Skip directories
        string image_name = ep->d_name;
        Mat I0;
        if(argv[2] == "sgm") {
            const auto t1 = std::chrono::system_clock::now();
            sgmComputeDisp(image_name, D0, D1);
            const auto t2 = std::chrono::system_clock::now();
            const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
            std::cout << "disparity computation time: " << duration << "[msec]" << std::endl;
        } else {
            I0 = depthComputeDisp(image_name, D0, coordinateTransform);
        }
        

        // 视差输出
        D0.convertTo(draw, CV_8U, 255. / (SemiGlobalMatching::DISP_SCALE * param.numDisparities));
        cv::applyColorMap(draw, draw, cv::COLORMAP_JET);   
        draw.setTo(0, D0 == SemiGlobalMatching::DISP_INV);
        Mat fdisp;
        D0.convertTo(fdisp, CV_32F, 1. / SemiGlobalMatching::DISP_SCALE);
        string dispName(image_name.begin(), image_name.end()-4);
        outputImg(fdisp, stixelDir, dispName, 1);
        
        cout << "D0 channels " << D0.channels() << endl;
        cout << "fdisp channels " << fdisp.channels() << endl;


        // 计算stixels
        std::vector<Stixel> stixels;
        std::vector<std::vector<int>> bboxes;
        const auto t3 = std::chrono::system_clock::now();
        StixelWorld::Parameters stix_param;
	    const cv::FileStorage cvfs(argv[2], FileStorage::READ);
        bool opened = cvfs.isOpened();
        StixelWorld stixelWorld(stix_param);

        stixelWorld.compute(fdisp, stixels, bboxes);
        const auto t4 = std::chrono::system_clock::now();
        const auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count();
        cout << "stixels num: " << stixels.size() << endl;
        cout << endl << " yes" << endl;
        std::cout << "stixel computation time: " << 1e-3 * duration2 << "[msec]" << std::endl;


        // 写入stixels
        cv::Mat showStixel = I0.clone();
        cv::Mat stixelImg = cv::Mat::zeros(I0.size(), showStixel.type());
        for (const auto& stixel : stixels)
            drawStixel(stixelImg, stixel, dispToColor(stixel.disp, (float)param.numDisparities));
		showStixel = showStixel + 0.5 * stixelImg;
        outputImg(showStixel, "../data/testForTiff/smgStixel/", image_name);

        // 写入bbox
        for(auto &bbox : bboxes)
        {
            // 0 3 2 1 是 左上角坐标点x1 y1 和右下角坐标点 x2 y2 的坐标
            cv::rectangle(showStixel, Point(bbox[0], bbox[3]), Point(bbox[2], bbox[1]), Scalar(0,0,255), 2, 8, 0);
            bboxFile << "(" << bbox[0] << "," << bbox[3] << ")" << endl;
            bboxFile << "(" << bbox[2] << "," << bbox[1] << ")" << endl;
            bboxFile.flush();
        }
    }
    bboxFile.close();
    return 0; 
}
