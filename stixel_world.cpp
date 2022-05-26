#include "stixel_world.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif


using namespace std;

struct Line
{
    Line(float a = 0, float b = 0) : a(a), b(b) {}
    Line(const cv::Point2f& pt1, const cv::Point2f& pt2)
    {
        a = (pt2.y - pt1.y) / (pt2.x - pt1.x);
        b = -a * pt1.x + pt1.y;
    }
    float a, b;
};


// Implementation of free space computation
class FreeSpace
{
    public:

        struct Parameters
        {
            float alpha1;       //* 物体权重
            float alpha2;       //* 道路权重
            float objectHeight; //* 假设的物体高度
            float Cs;           //* cost parameter penalizing jumps in depth 成本参数惩罚深度跳跃, 用来计算freeplace边界路径
            float Ts;           //* threshold saturating the cost function  使成本函数饱和的阈值, 用来计算freeplace边界路径
            int maxPixelJump;   //* 一次最大的跳数
            int mode;
            // default settings
            Parameters()
            {
                alpha1 = 2;
                alpha2 = 1;     // ! 这个权重应该是需要调整的, 因为我们更关注地上的不平整
                objectHeight = 0.5f;    
                Cs = 50;
                Ts = 32;
                maxPixelJump = 100;
            }
        };

        FreeSpace(const Parameters& param = Parameters()) : param_(param)
    {
    }
        // 传进来的只是视差, 可以猜想关于占用网格的计算包含在这个函数其中
        void compute(
                    const cv::Mat1f& disparity, std::vector<float>& roadDisp,       //! 这两个是输入的视差数据, 为什么有两个视差 
                    int vhor, 
                    std::vector<int>& path, std::vector<int>& oriPath,    // 这两个path是返回结果的
                    const CameraParameters& camera)
        {
            const int umax = disparity.rows;
            const int vmax = disparity.cols;

            //*  这两个Mat是计算结果存储
            cv::Mat1f score(umax, vmax, std::numeric_limits<float>::max());  // 1, one channel, 不是一维;
                                                                             // 很明显, 这个score就是占用网格数据
            cv::Mat1i table(umax, vmax, 0);                                  // i, int
                                                                        
            CoordinateTransform tf(camera);

            /////////////////////////////////////////////////////////////////////////////
            // compute score image for the free space
            //////////////////////////////////////////////////////////////////////////////
            const float SCORE_DEFAULT = 1.f;

            int u;
            // #pragma omp parallel for
            std::vector<int> *vT;

            for (u = 0; u < umax; u++)
            {
                // compute and accumlate differences between measured disparity and expected road disparity
                // 对, 没错, 公式里需要一个现实d 和一个理论计算d, 看公式和代码对应就行, 公式具体怎么推导, 我也不会
                std::vector<float> integralRoadDiff(vmax);
                float tmpSum = 0.f;

                for (int v = vhor; v < vmax; v++)
                {
                    const float roadDiff = disparity(u, v) > 0.f ? fabsf(disparity(u, v) - roadDisp[v]) : SCORE_DEFAULT;    // 理论与实际视差差
                    tmpSum += roadDiff;     // 按列叠加的理论与实际视差差的和
                    integralRoadDiff[v] = tmpSum;   // 存放每一列tmpSum
                }

                // compute search range
                // ! 涉及图形计算, 还没弄懂
                vT = new std::vector<int>(vmax, 0);
                for (int vB = vhor; vB < vmax; vB++)
                {
                    const float YB = tf.toY(roadDisp[vB], vB);
                    const float ZB = tf.toZ(roadDisp[vB], vB);
                    const float YT = YB - param_.objectHeight;
                    (*vT)[vB] = std::max(cvRound(tf.toV(YT, ZB)), 0); // 计算的V的点位置; cvRound四舍五入
                }

                for (int vB = vhor; vB < vmax; vB++)
                {
                    // compute the object score
                    float objectScore = 0.f;
                    for (int v = (*vT)[vB]; v < vB; ++v)    // 从计算的v点位置到实际的v点位置都累加在objectScore上
                        // objectScore也是通过roadDisp计算出, 所以roadDisp不能删
                        objectScore += disparity(u, v) > 0.f ? fabsf(disparity(u, v) - roadDisp[vB]) : SCORE_DEFAULT;   
                        // objectScore += 0;   

                    // compute the road score
                    // ! 没懂roadScore
                    const float roadScore = integralRoadDiff[vmax - 1] - integralRoadDiff[vB - 1];   // 从vB列到最后一列的
                    score(u, vB) = (float)(param_.alpha1 * objectScore + 1 * roadScore);
                    // const float roadScore = 0;
                    // score(u, vB) = (float)(3 * objectScore + param_.alpha2 * roadScore);
                }
                delete vT;
            }

#if 0
            oriPath.resize(umax);
            //Test to see original score
            for(int m=0; m<umax; m++)
            {
                float minScore = std::numeric_limits<float>::max();
                int minv = -1;
                for (int vB = vhor; vB < vmax; vB++)
                {
                    // compute the object score
                    float fScore = score.at<float>(m, vB);

                    if (fScore < minScore)
                    {
                        minScore = fScore;
                        minv = vB;
                    }
                }
                oriPath[m] = minv;
            }
#endif

            /////////////////////////////////////////////////////////////////////////////
            // extract the optimal free space path by dynamic programming
            //////////////////////////////////////////////////////////////////////////////
            // forward step
            for (int uc = 1; uc < umax; uc++)
            {
                const int up = uc - 1;      // up 是uc 的前一行, 而不是同一行

                int vc;
#pragma omp parallel for
                for (vc = vhor; vc < vmax; vc++)
                {
                    // 搜索范围[vp1, vp2], 基本上是全局搜索
                    //! 如果要优化计算速度可以减少搜索范围入手
                    const int vp1 = std::max(vc - param_.maxPixelJump, vhor);   // 不过maxPixelJump时使用vhor
                    const int vp2 = std::min(vc + param_.maxPixelJump + 1, vmax);   // 仅差maxPixelJump时使用vmax

                    float minScore = std::numeric_limits<float>::max();
                    int minv = 0;
                    for (int vp = vp1; vp < vp2; vp++)  // 在搜索范围内寻找惩罚最低的下一跳
                    {
                        const float dc = disparity(uc, vc);     // 当前点的视差
                        const float dp = disparity(up, vp);     // 下一跳的视差(未确认)
                        const float dispJump = (dc >= 0.f && dp >= 0.f) ? fabsf(dp - dc) : SCORE_DEFAULT;   // 越过的视差值
                        const float penalty = std::min(param_.Cs * dispJump, param_.Cs * param_.Ts);
                        const float s = score(up, vp) + penalty;
                        if (s < minScore)
                        {
                            minScore = s;
                            minv = vp;
                        }
                    }

                    score(uc, vc) += minScore;  // 累加下一跳的代价
                    table(uc, vc) = minv;       // 保存当前点的下一跳
                }
            }

            // backward step
            path.resize(umax);
            float minScore = std::numeric_limits<float>::max();
            int minv = 0;
            for (int v = vhor; v < vmax; v++)  // 在最大行找最小的minScore
            {
                if (score(umax - 1, v) < minScore)
                {
                    minScore = score(umax - 1, v);
                    minv = v;
                }
            }
            for (int u = umax - 1; u >= 0; u--) // 循环获得下一跳
            {
                path[u] = minv;
                minv = table(u, minv);
            }
        }

    private:
        Parameters param_;
};

// Implementation of height segmentation
class HeightSegmentation
{
    public:

        struct Parameters
        {
            float deltaZ;     //!< allowed deviation in [m] to the base point
            float Cs;         //!< cost parameter penalizing jumps in depth and pixel
            float Nz;         //!< if the difference in depth between the columns is equal or larger than this value, cost of a jump becomes zero
            int maxPixelJump; //!< maximum allowed jumps in pixel (higher value increases computation time)

            // default settings
            Parameters()
            {
                deltaZ = 5;
                Cs = 8;
                Nz = 5;
                maxPixelJump = 50;
            }
        };

        HeightSegmentation(const Parameters& param = Parameters()) : param_(param)
    {
    }

        void compute(const cv::Mat1f& disparity, const std::vector<int>& lowerPath, std::vector<int>& upperPath, const CameraParameters& camera)
        {
            const int umax = disparity.rows;
            const int vmax = disparity.cols;

            cv::Mat1f score(umax, vmax, std::numeric_limits<float>::max());
            cv::Mat1i table(umax, vmax, 0);

            CoordinateTransform tf(camera);

            /////////////////////////////////////////////////////////////////////////////
            // compute score image for the height segmentation
            //////////////////////////////////////////////////////////////////////////////
            int u;
#pragma omp parallel for
            for (u = 0; u < umax; u++)
            {
                // get the base point
                const int vB = lowerPath[u];
                const float dB = disparity(u, vB);

                // deltaD represents the allowed deviation in disparity
                float deltaD = 0.f;
                if (dB > 0.f)
                {
                    const float YB = tf.toY(dB, vB);
                    const float ZB = tf.toZ(dB, vB);
                    deltaD = dB - tf.toD(YB, ZB + param_.deltaZ);
                }

                // compute and accumlate membership value
                std::vector<float> integralMembership(vmax);
                float tmpSum = 0.f;
                for (int v = 0; v < vmax; v++)
                {
                    const float d = disparity(u, v);

                    float membership = 0.f;
                    if (dB > 0.f && d > 0.f)
                    {
                        const float deltad = (d - dB) / deltaD;         // 计算当前点和该行的base点的差值 (除一个和dB相关的固定东西应该是为了归一化)
                        const float exponent = 1.f - deltad * deltad;   // 指数
                        membership = powf(2.f, exponent) - 1.f;         // 用指数起到放大的作用
                    }

                    tmpSum += membership;                               // menbership理解成经过处理的视差差, 用tmpSum叠加
                    integralMembership[v] = tmpSum; 
                }

                score(u, 0) = integralMembership[vB - 1];
                for (int vT = 1; vT < vB; vT++)
                {
                    const float score1 = integralMembership[vT - 1];
                    const float score2 = integralMembership[vB - 1] - integralMembership[vT - 1];
                    score(u, vT) = score1 - score2;
                }
            }

            /////////////////////////////////////////////////////////////////////////////
            // extract the optimal height path by dynamic programming
            //////////////////////////////////////////////////////////////////////////////
            // forward step
            for (int uc = 1; uc < umax; uc++)
            {
                const int up = uc - 1;
                const int vB = lowerPath[uc];

                int vc;
#pragma omp parallel for
                for (vc = 0; vc < vB; vc++)
                {
                    const int vp1 = std::max(vc - param_.maxPixelJump, 0);
                    const int vp2 = std::min(vc + param_.maxPixelJump + 1, vB);

                    float minScore = std::numeric_limits<float>::max();
                    int minv = 0;
                    for (int vp = vp1; vp < vp2; vp++)
                    {
                        const float dc = disparity(uc, vc);
                        const float dp = disparity(up, vp);

                        float Cz = 1.f;
                        if (dc > 0.f && dp > 0.f)
                        {
                            const float Zc = tf.toZ(dc, vc);
                            const float Zp = tf.toZ(dp, vp);
                            Cz = std::max(0.f, 1 - fabsf(Zc - Zp) / param_.Nz);
                        }

                        const float penalty = param_.Cs * abs(vc - vp) * Cz;
                        const float s = score(up, vp) + penalty;
                        if (s < minScore)
                        {
                            minScore = s;
                            minv = vp;
                        }
                    }

                    score(uc, vc) += minScore;
                    table(uc, vc) = minv;
                }
            }

            // backward step
            upperPath.resize(umax);
            float minScore = std::numeric_limits<float>::max();
            int minv = 0;
            for (int v = 0; v < vmax; v++)
            {
                if (score(umax - 1, v) < minScore)
                {
                    minScore = score(umax - 1, v);
                    minv = v;
                }
            }
            for (int u = umax - 1; u >= 0; u--)
            {
                upperPath[u] = minv;
                minv = table(u, minv);
            }
        }

    private:
        Parameters param_;
};

// estimate road model from camera tilt and height
static Line calcRoadModelCamera(const CameraParameters& camera)
{
    const float sinTilt = sinf(camera.tilt);
    const float cosTilt = cosf(camera.tilt);
    const float a = (camera.baseline / camera.height) * cosTilt;
    const float b = (camera.baseline / camera.height) * (camera.fu * sinTilt - camera.v0 * cosTilt);
    return Line(a, b);
}

// estimate road model from v-disparity
static Line calcRoadModelVD(const cv::Mat1f& disparity, const CameraParameters& camera,
        int samplingStep = 2, int minDisparity = 10, int maxIterations = 32, float inlierRadius = 1, float maxCameraHeight = 5)
{
    const int umax = disparity.rows;
    const int vmax = disparity.cols;

    // sample v-disparity points
    std::vector<cv::Point2f> points;
    points.reserve(vmax * umax);
    for (int u = 0; u < umax; u += samplingStep)
        for (int v = 0; v < vmax; v += samplingStep)
            if (disparity(u, v) >= minDisparity)
                points.push_back(cv::Point2f(static_cast<float>(v), disparity(u, v)));

    if (points.empty())
        return Line(0, 0);

    // estimate line by RANSAC
    cv::RNG random;
    Line bestLine;
    int maxInliers = 0;
    for (int iter = 0; iter < maxIterations; iter++)
    {
        // sample 2 points and get line parameters
        const cv::Point2f& pt1 = points[random.next() % points.size()];
        const cv::Point2f& pt2 = points[random.next() % points.size()];
        if (pt1.x == pt2.x)
            continue;

        const Line line(pt1, pt2);

        // estimate camera tilt and height
        const float tilt = atanf((line.a * camera.v0 + line.b) / (camera.fu * line.a));
        const float height = camera.baseline * cosf(tilt) / line.a;

        // skip if not within valid range
        if (height <= 0.f || height > maxCameraHeight)
            continue;

        // count inliers within a radius and update the best line
        int inliers = 0;
        for (const auto& pt : points)
            if (fabs(line.a * pt.x + line.b - pt.y) <= inlierRadius)
                inliers++;

        if (inliers > maxInliers)
        {
            maxInliers = inliers;
            bestLine = line;
        }
    }

    // apply least squares fitting using inliers around the best line
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    int n = 0;
    for (const auto& pt : points)
    {
        const float x = pt.x;
        const float y = pt.y;
        const float yhat = bestLine.a * x + bestLine.b;
        if (fabs(yhat - y) <= inlierRadius)
        {
            sx += x;
            sy += y;
            sxx += x * x;
            syy += y * y;
            sxy += x * y;
            n++;
        }
    }

    const float a = static_cast<float>((n * sxy - sx * sy) / (n * sxx - sx * sx));
    const float b = static_cast<float>((sxx * sy - sxy * sx) / (n * sxx - sx * sx));
    return Line(a, b);
}

static float calcAverageDisparity(const cv::Mat& disparity, const cv::Rect& rect, int minDisp, int maxDisp)
{
    const cv::Mat dispROI = disparity(rect & cv::Rect(0, 0, disparity.cols, disparity.rows));
    const int histSize[] = { maxDisp - minDisp };
    const float range[] = { static_cast<float>(minDisp), static_cast<float>(maxDisp) };
    const float* ranges[] = { range };

    cv::Mat hist;
    cv::calcHist(&dispROI, 1, 0, cv::Mat(), hist, 1, histSize, ranges);

    int maxIdx[2];
    cv::minMaxIdx(hist, NULL, NULL, NULL, maxIdx);

    return (range[1] - range[0]) * maxIdx[0] / histSize[0] + range[0];
}

StixelWorld::StixelWorld(const Parameters & param) : param_(param)
{
}


// 计算stixels的总函数
void StixelWorld::compute(const cv::Mat& disparity, std::vector<Stixel>& stixels, std::vector<std::vector<int>>& bboxes)
{
    CV_Assert(disparity.type() == CV_32F);
    CV_Assert(param_.stixelWidth % 2 == 1);

    const int stixelWidth = param_.stixelWidth;
    const int umax = disparity.cols / stixelWidth;  // * 指的是stixel的列数而不是像素的列数
    const int vmax = disparity.rows;
    CameraParameters camera = param_.camera;

    // compute horizontal median of each column
    cv::Mat1f columns(umax, vmax);  // umax用的列, vmax用的行, 相当于把图像沿y=x对折
    std::vector<float> buf(stixelWidth);

    // * 先把图像压缩，水平7个压缩成一个，用其中位数
    for (int v = 0; v < vmax; v++)
    {
        for (int u = 0; u < umax; u++)
        {
            // compute horizontal median
            for (int du = 0; du < stixelWidth; du++)
                buf[du] = disparity.at<float>(v, u * stixelWidth + du);
            std::sort(std::begin(buf), std::end(buf));
            const float m = buf[stixelWidth / 2];

            // store with transposed
            columns.ptr<float>(u)[v] = m;
        }
    }
    // cv::imshow("error", columns);
    
    // compute road model (assumes planar surface)
    Line line;
    if (param_.roadEstimation == ROAD_ESTIMATION_AUTO)
    {
        line = calcRoadModelVD(columns, camera);

        // when AUTO mode, update camera tilt and height
        camera.tilt = atanf((line.a * camera.v0 + line.b) / (camera.fu * line.a));
        camera.height = camera.baseline * cosf(camera.tilt) / line.a;

        std::cout<<"(tilt, height): "<<camera.tilt<<","<<camera.height<<std::endl;
    }
    else if (param_.roadEstimation == ROAD_ESTIMATION_CAMERA)
    {
        line = calcRoadModelCamera(camera);
    }
    else
    {
        CV_Error(cv::Error::StsInternal, "No such mode");
    }

    // compute expected road disparity
    // 根据几何关系可以计算出平面道路的理论视差 (虽然我并不知道他是怎么算的)
    std::vector<float> roadDisp(vmax);
    for (int v = 0; v < vmax; v++)
    {
        roadDisp[v] = line.a * v + line.b;
    }

    // horizontal row from which road disparity becomes negative
    const int vhor = abs(cvRound(-line.b / line.a));

    FreeSpace freeSpace;
    std::vector<int> oPath;
    freeSpace.compute(columns, roadDisp, vhor, lowerPath_, oPath, camera);          
    HeightSegmentation heightSegmentation;
    heightSegmentation.compute(columns, lowerPath_, upperPath_, camera);

    // extract disparity
    std::cout << "stixels num " << stixels.size() << std::endl;
    stixels.clear();
    for (int u = 0; u < umax; u++)
    {
        const int vT = upperPath_[u];
        const int vB = lowerPath_[u];
        const int stixelHeight = vB - vT;
        const cv::Rect stixelRegion(stixelWidth * u, vT, stixelWidth, stixelHeight);

        Stixel stixel;
        stixel.u = stixelWidth * u + stixelWidth / 2;
        stixel.vT = vT;
        stixel.vB = vB;
        stixel.width = stixelWidth;
        stixel.disp = calcAverageDisparity(disparity, stixelRegion, param_.minDisparity, param_.maxDisparity);
        stixels.push_back(stixel);
        // std::cout << "stixels num " << stixels.size() << std::endl;
    }

    std::vector<std::vector<Stixel>> objects;
    double oriDifferRadio = vmax*0.05;    // 棒状像素起点在20上下将加入同一个框
    for(int i = 0; i < stixels.size(); i++)
    {
        
        std::vector<Stixel> object;
        double meanVB = stixels[i].vB;
        double stixelLen = abs(stixels[i].vT - stixels[i].vB);
        double meanLen = stixelLen;
        double differRadio = stixelLen*0.25 < oriDifferRadio ? stixelLen*0.1 : oriDifferRadio;
        while(i < stixels.size()
              && abs(stixels[i].vB - meanVB) <= differRadio 
            //   && abs(abs(stixels[i].vT - stixels[i].vB) - meanLen) < meanLen
              )
        {
            object.push_back(stixels[i]);
            meanVB = (meanVB*(object.size()-1) + stixels[i].vB)/object.size();
            // meanLen = (meanLen*(object.size()-1) + abs(stixels[i].vT - stixels[i].vB))/object.size();
            i++;
        }
        objects.push_back(object);
    }
    
    for(auto &object : objects)
    {
        std::vector<int> bbox;
        bbox.push_back(object[0].u); // X1
        // double minVB = INT_MAX;
        double sumVB = 0;
        for(auto &stixel : object)
        {
            sumVB += stixel.vB;
        }
        bbox.push_back(sumVB/object.size());  // Y2
        bbox.push_back(object[object.size()-1].u); // X2
        std::sort(object.begin(), object.end(), [](Stixel s1, Stixel s2){ return s1.vT < s2.vT; });
        bbox.push_back(object[object.size()/2].vT); // Y1
        bboxes.push_back(bbox);
    }
}

void StixelWorld::computeDepth(const cv::Mat& disparity, cv::Mat& depthMat)
{
    cout << "disparity channels: " << disparity.channels() << endl;
    int height = disparity.rows;
    int width = disparity.cols;

    CameraParameters camera = this->param_.camera;
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            int id = i*width + j;
            if(!disparity.at<float>(i, j))
                continue;
            depthMat.at<float>(i,j) = camera.fu * camera.baseline / disparity.at<float>(i, j);
            // if(j == width/2)
            // {
                // std::cout << depthMatData[id] << std::endl;
            // }
        }
    }
}

void StixelWorld::computeDisp(const cv::Mat& disparity, cv::Mat& depthMat) {
    
}
