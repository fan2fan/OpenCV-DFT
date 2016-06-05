#ifndef FFT
#define FFT

#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>

using namespace cv;
using namespace std;

#define DFT 0
#define IDFT 1

//对矩阵进行傅里叶变换，通过isNonBorder参数实现是否加边
void fft(Mat& inputArray,Mat& outputArray,bool isNonBorder)
{
    Mat padded,dftData;
    //如果不加边isNonBorder = true
    if(isNonBorder)
    {
        padded = inputArray.clone();
    }
    //加边则：
    else
    {
        int borderRows = getOptimalDFTSize(inputArray.rows);
        int borderCols = getOptimalDFTSize(inputArray.cols);
        copyMakeBorder(inputArray,padded,0,borderRows-inputArray.rows,0,
                       borderCols-inputArray.cols,BORDER_CONSTANT,
                       Scalar::all(0));
    }
    //planes[2]用来保存实部和虚部
    Mat planes[2] = {Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F)};
    merge(planes,2,dftData);
    dft(dftData,dftData);

    outputArray = dftData.clone();
}

//进行移位
void fftShift(Mat& inputArray, Mat& outputArray)
{

    Mat shiftArray;
    //剪切和重分布幅度图像限,下面是使行和列的数目都为偶数
    shiftArray = inputArray(Rect(0,0,inputArray.cols&-2,inputArray.rows&-2));
    //    shiftArray = inputArray(Range(0,inputArray.rows&-2),Range(0,inputArray.cols&-2)).clone();
    //如果要进行平移的话
    int cols = shiftArray.cols;
    int rows = shiftArray.rows;
    int cx = cols/2;
    int cy = rows/2;
    //q0,q1,q2,q3的变换直接导致shiftArray的改变
    Mat q0(shiftArray,Rect(0,0,cx,cy));
    Mat q1(shiftArray,Rect(cx,0,cx,cy));
    Mat q2(shiftArray,Rect(0,cy,cx,cy));
    Mat q3(shiftArray,Rect(cx,cy,cx,cy));
    //    交换象限
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    outputArray = shiftArray.clone();
}
//求傅里叶矩阵的幅值
void fftMagnitude(Mat& inputArray, Mat& outputArray)
{
    Mat planes[2];
    Mat dftData = inputArray.clone();
    //将傅里叶变换后的矩阵拆分为虚部和实部

    split(dftData,planes);
    //取矩阵的幅值
    magnitude(planes[0],planes[1],planes[0]);
    outputArray = planes[0].clone();
}


//对傅里叶变换后的矩阵进行逆变换
void ifft(Mat& inputArray,Mat& outputArray)
{
    Mat dftData = inputArray.clone();
    Mat planes[2];
    dft(dftData,dftData,DFT_INVERSE);
    split(dftData,planes);
    //只取实部
    outputArray = planes[0].clone();
}


//将傅里叶变换或者傅里叶逆变换的矩阵进行映射到0-255灰度空间
void mappedDFT(Mat& inputArray,Mat& outputArray,int dftMethod,bool isNormalize)
{
    Mat mappedArray = inputArray.clone();
    if(dftMethod==0)
    {
        mappedArray += Scalar::all(1);
        log(mappedArray,mappedArray);
    }
    //如果进行归一化
    if(isNormalize)
        normalize(mappedArray,mappedArray,0,1,NORM_MINMAX);

    //这里通过最大最小值来检验移位是否成功，不需要可以将他屏蔽掉
    double minVal,maxVal;
    Point minIdx,maxIdx;
    //输出矩阵的最大值最小值以及其索引（下标）
    minMaxLoc(mappedArray,&minVal,&maxVal,&minIdx,&maxIdx,noArray());
    cout<<"The minimum and the maximum: "<<minVal<<"  "<<maxVal<<endl;
    cout<<"And their indexes are:       "<<minIdx<<"  "<<maxIdx<<endl;
    cout<<endl<<endl<<endl;
    //输出
    outputArray = mappedArray.clone();
}

/*关于矩阵赋值问题：
 * opencv矩阵赋值函数copyTo、clone、重载元算符 ‘=’之间实现的功能相似均是给不同的矩阵赋值功能。
 * copyTo和clone函数基本相同，被赋值的矩阵和赋值矩阵之间空间独立，不共享同一空间。
 * 但是重载元算赋‘=’，被赋值的矩阵和赋值矩阵之间空间共享，改变任一个矩阵的值，会同时影响到另一个矩阵。
 * 当矩阵作为函数的返回值时其功能和重载元算赋‘=’相同.
 * 所以我们在对矩阵赋值时优先使用clone和copyTo
 * */

#endif // FFT

