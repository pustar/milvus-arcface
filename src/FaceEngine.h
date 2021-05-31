#pragma once
//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//虹软人脸识别
#include "arcsoft_face_sdk.h"
#include "amcomdef.h"
#include "asvloffscreen.h"

#include <iostream>
#include <string>
#include <Windows.h>
#include <fstream>
#include <vector>


using namespace std;
using namespace cv;

class FaceEngine
{
public:
	FaceEngine(ASF_DetectMode detect_mode, ASF_OrientPriority op, int max_face_num);
	~FaceEngine();

private:
	FaceEngine(const FaceEngine&) = delete;
	FaceEngine& operator=(const FaceEngine&) = delete;


public:
	ASF_MultiFaceInfo  DetectFace(Mat& img);
	ASF_SingleFaceInfo GetSingleFace(ASF_MultiFaceInfo& faces, int i);
	ASF_FaceFeature GetFaceFeature(Mat& image, ASF_SingleFaceInfo& face, bool deepCopy, int hasMask, bool isReg);
	float FaceEngine::FaceCompare(ASF_FaceFeature& face1, ASF_FaceFeature& face2);
	bool FaceEngine::IsRGBLive(Mat& image, ASF_SingleFaceInfo& face);
	bool SaveFaceFeature(ASF_FaceFeature& face, string file);
	ASF_FaceFeature LoadFaceFeature(string file);
	void DrawFaceRect(Mat& img, ASF_SingleFaceInfo face, int shape, Scalar color, int thickness);
	void DrawFaceRect(Mat& img, ASF_MultiFaceInfo faces, int shape, Scalar color, int thickness);

private:
	MHandle EngineHandle;
	Mat ImageProcess(Mat& img);

};

FaceEngine::FaceEngine(ASF_DetectMode detect_mode, ASF_OrientPriority op, int max_face_num)
{
	if (max_face_num > 10) max_face_num = 10;

	int res = ASFInitEngine(detect_mode, op, max_face_num,
		ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_LIVENESS, &EngineHandle);
	if (res != 0) throw res;
	ASF_LivenessThreshold ts = { 0 };
	ts.thresholdmodel_BGR = 0.75;
	ASFSetLivenessParam(EngineHandle, &ts);
}

FaceEngine::~FaceEngine()
{
	ASFUninitEngine(EngineHandle);
}

Mat FaceEngine::ImageProcess(Mat& img)
{
	if (img.cols % 4 != 0)
	{
		Mat tmp = img(Rect(0, 0, img.cols - (img.cols % 4), img.rows)).clone();
		return tmp;
	}
	return img;
}

ASF_SingleFaceInfo FaceEngine::GetSingleFace(ASF_MultiFaceInfo& faces, int i)
{
	ASF_SingleFaceInfo face = { 0 };
	face.faceRect.left = faces.faceRect[i].left;
	face.faceRect.top = faces.faceRect[i].top;
	face.faceRect.right = faces.faceRect[i].right;
	face.faceRect.bottom = faces.faceRect[i].bottom;
	face.faceOrient = faces.faceOrient[i];
	face.faceDataInfo = faces.faceDataInfoList[i];
	return face;
}

// 注意！DetectFace都会覆盖上次的结果。
ASF_MultiFaceInfo FaceEngine::DetectFace(Mat& image)
{
	Mat img = ImageProcess(image);
	ASF_MultiFaceInfo faces = { 0 };
	int res = ASFDetectFaces(EngineHandle, img.cols, img.rows, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)img.data, &faces);
	if (res != 0) throw res;
	//for (int i = 0; i < faces.faceNum; i++)
	//{
	//	void * dataAddress = faces.faceDataInfoList[i].data;
	//	faces.faceDataInfoList[i].data = malloc(faces.faceDataInfoList[i].dataSize);
	//	memcpy(faces.faceDataInfoList[i].data, dataAddress, faces.faceDataInfoList[i].dataSize);
	//}

	return faces;
}

ASF_FaceFeature FaceEngine::GetFaceFeature(Mat& image, ASF_SingleFaceInfo& face, bool deepCopy = false, int hasMask = 0, bool isReg = false)
{
	Mat img = ImageProcess(image);
	ASF_FaceFeature fea = { 0 };
	int res = ASFFaceFeatureExtract(EngineHandle, img.cols, img.rows, ASVL_PAF_RGB24_B8G8R8,
		(MUInt8*)img.data, &face, isReg ? ASF_REGISTER : ASF_RECOGNITION, hasMask, &fea);
	if (res != 0) throw res;
	if (deepCopy)
	{
		void* t = fea.feature;
		fea.feature = (byte*)malloc(fea.featureSize);
		memcpy(fea.feature, t, fea.featureSize);
	}
	return fea;
}

float  FaceEngine::FaceCompare(ASF_FaceFeature& face1, ASF_FaceFeature& face2)
{
	float res = 0;
	int resc = ASFFaceFeatureCompare(EngineHandle, &face1, &face2, &res);
	if (resc != 0) throw resc;
	return res;
}

bool FaceEngine::IsRGBLive(Mat& image, ASF_SingleFaceInfo& face)
{
	Mat img = ImageProcess(image);

	ASF_MultiFaceInfo faces = { 0 };
	faces.faceRect = &(face.faceRect);
	faces.faceOrient = &(face.faceOrient);
	faces.faceDataInfoList = &(face.faceDataInfo);
	faces.faceNum = 1;
	int res;
	res = ASFProcess(EngineHandle, img.cols, img.rows, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)img.data, &faces, ASF_LIVENESS);
	if (res != 0) throw res;
	ASF_LivenessInfo info = { 0 };
	res = ASFGetLivenessScore(EngineHandle, &info);
	if (res != 0) throw res;
	return info.isLive[0] == 1;
}

bool FaceEngine::SaveFaceFeature(ASF_FaceFeature& face, string file)
{
	fstream fs(file, ios_base::out | ios_base::binary);
	if (!fs.is_open()) return false;
	fs.write((const char*)face.feature, face.featureSize);
	fs.close();
	return true;
}

ASF_FaceFeature FaceEngine::LoadFaceFeature(string file)
{
	ASF_FaceFeature fea = { 0 };
	fstream fs(file, ios_base::in | ios_base::binary);
	if (fs.is_open())
	{
		fea.featureSize = 2056;
		fea.feature = (byte*)malloc(2056);
		fs.read((char*)fea.feature, 2056);
	}
	return fea;
}

// shape = 1 , 正方形
// shape != 1 , 圆形
void FaceEngine::DrawFaceRect(Mat& img, ASF_SingleFaceInfo face, int shape = 1, Scalar color = Scalar(0, 255, 0), int thickness = 2)
{
	Rect rect(
		face.faceRect.left,
		face.faceRect.top,
		face.faceRect.right - face.faceRect.left,
		face.faceRect.bottom - face.faceRect.top
	);
	if (shape == 1)
	{
		rectangle(img, rect, color, thickness);
	}
	else
	{
		circle(img, Point(rect.x + rect.width / 2, rect.y + rect.height / 2), (int)(rect.width / 1.6), color, thickness);
	}
}

// shape = 1 , 正方形
// shape != 1 , 圆形
void FaceEngine::DrawFaceRect(Mat& img, ASF_MultiFaceInfo faces, int shape = 1, Scalar color = Scalar(0, 255, 0), int thickness = 2)
{
	for (int i = 0; i < faces.faceNum; i++)
	{
		auto face = GetSingleFace(faces, i);
		DrawFaceRect(img, face, shape, color, thickness);
	}
}

