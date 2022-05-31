#ifndef __STIXEL_WORLD_H__
#define __STIXEL_WORLD_H__

#include <opencv2/opencv.hpp>


using CameraParameters = StixelWorld::CameraParameters;
using namespace std;

/** @brief Stixel struct
*/
struct Stixel
{
	int u;                        //*< stixel center x position
	int vT;                       //*< stixel top y position
	int vB;                       //*< stixel bottom y position
	int width;                    //*< stixel width
	float disp;                   //*< stixel average disparity
};

/** @brief StixelWorld class.

The class implements the static Stixel computation based on [1,2].
[1] D. Pfeiffer, U. Franke: "Efficient Representation of Traffic Scenes by means of Dynamic Stixels"
[2] H. Badino, U. Franke, and D. Pfeiffer, "The stixel world - a compact medium level representation of the 3d-world,"
*/

struct CoordinateTransform
{
    CoordinateTransform(const CameraParameters& camera) : camera(camera)
    {
        sinTilt = (sinf(camera.tilt));
        cosTilt = (cosf(camera.tilt));
        B = camera.baseline * camera.fu / camera.fv;
    }

    inline float toY(float d, int v) const
    {
        return (B / d) * ((v - camera.v0) * cosTilt + camera.fv * sinTilt);
    }

    inline float toZ(float d, int v) const
    {
        return (B / d) * (camera.fv * cosTilt - (v - camera.v0) * sinTilt);
    }

    inline float toV(float Y, float Z) const
    {
        return camera.fv * (Y * cosTilt - Z * sinTilt) / (Y * sinTilt + Z * cosTilt) + camera.v0;
    }

    inline float toD(float Y, float Z) const
    {
        return camera.baseline * camera.fu / (Y * sinTilt + Z * cosTilt);
    }

    CameraParameters camera;
    float sinTilt, cosTilt, B;
};

class StixelWorld
{
public:

	enum
	{
                                  // * 这两个应该是用于不平整的道路检测的参数
		ROAD_ESTIMATION_AUTO = 0, //*< road disparity are estimated by input disparity
		ROAD_ESTIMATION_CAMERA    //*< road disparity are estimated by camera tilt and height
	};

	/** @brief CameraParameters struct
	*/
	struct CameraParameters
	{
        // 这些会从camera.xml文件中读出
		float fu;                 //* x轴的焦距
		float fv;                 //* y轴的焦距
		float u0;                 //* 镜头中心点, 主点x
		float v0;                 //* 主点y
		float baseline;           //* 双目相机的基线
		float height;             //* 相机水平高度
		float tilt;               //* 相机倾斜角

		// default settings
		CameraParameters()
		{
			fu = 1.f;
			fv = 1.f;
			u0 = 0.f;
			v0 = 0.f;
			baseline = 0.2f;
			height = 1.f;
			tilt = 0.f;
		}
	};

	/** @brief Parameters struct
	*/
	struct Parameters
	{
		int stixelWidth;          //* stixel的像素宽度, 必须为奇数
		int minDisparity;         //* 输入最小视差
		int maxDisparity;         //* 输入最大视差
		int roadEstimation;       //* 道路视差评估模型
		CameraParameters camera;  //* 相机参数

		// default settings
		Parameters()
		{
			stixelWidth = 7;
			minDisparity = -1;
			maxDisparity = 64;
			camera = CameraParameters();
			roadEstimation = ROAD_ESTIMATION_AUTO;
		}
	};


	StixelWorld(const Parameters& param = Parameters());

	/** @brief Computes stixels in a disparity map
	@param disparity 32-bit single-channel disparity map
	@param output array of stixels
	*/
	void compute(const cv::Mat& disparity, std::vector<Stixel>& stixels, std::vector<std::vector<int>>& bboxes);

    void computeDepth(const cv::Mat& disparity, cv::Mat& depthMat);
    
    
private:
	std::vector<int> lowerPath_, upperPath_;
	Parameters param_;
};

#endif // !__STIXEL_WORLD_H__