#include "FaceEngine.h"
#include <string>
#include <atomic>
#include <fstream>
#include "TP.cpp"
#include <windows.h>

atomic_int n = 0;

void task(int start, int end, int index)
{
	ofstream save("D:/Face/feature/" + to_string(index) + ".txt");
	//调用前请先确保已经激活
	FaceEngine x(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, 1);
	char file[50] = { 0 };
	for (int i = start; i <= end; i++)
	{
		sprintf(file, "D:/Face/img_celeba/%06d.jpg", i);
		Mat img = imread(file);
		auto faces = x.DetectFace(img);
		if (faces.faceNum == 1)
		{
			auto face = x.GetSingleFace(faces, 0);
			auto f = x.GetFaceFeature(img, face);
			float data[256];
			memcpy(data, f.feature + 8 + 1024, 1024);
			save << i << "|";
			for (int u = 0; u < 256; u++)
				save << data[u] << "|";
			save << endl;
		}
		n++;
	}

}

int main()
{
	ThreadPool pool(2);
	pool.AddTask(bind(task, 1, 100000, 1));
	pool.AddTask(bind(task, 100001, 202599, 2));

	while (n < 202599)
	{
		cout << "\r" << n << "\t" << n * 100 / 202599 << "\t";
		Sleep(1000);
	}
}
