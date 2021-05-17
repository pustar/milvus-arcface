
#pragma once
#include <vector>
#include <queue>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <windows.h>

using namespace std;

class ThreadPool
{
public:
	using Task = function<void()>;

private:
	int threadNum;
	atomic_int availableNum;
	mutex mtx;
	condition_variable cond;
	vector<thread> threads;
	queue<Task> tasks;

public:

	size_t availableThreads() { return availableNum; }
	size_t nowTasks() { return tasks.size(); }

	static int GetCPUCores()
	{
		SYSTEM_INFO info;
		GetSystemInfo(&info);
		return info.dwNumberOfProcessors;
	}

	ThreadPool(int nThread = 4)
	{
		threadNum = nThread;
		availableNum = nThread;

		for (int i = 0; i < threadNum; i++)
			threads.emplace_back(thread(&ThreadPool::TP_Work, this));
	}

	
	void AddTask(const Task& task)
	{
		lock_guard <mutex> lk(mtx);
		tasks.push(task);
		cond.notify_one();
	}

private:
	void TP_Work()
	{
		while (true)
		{
			Task task;
			{
				unique_lock<mutex> lk(mtx);
				if (!tasks.empty())
				{
					task = tasks.front();
					tasks.pop();
				}
				else if (tasks.empty())
					cond.wait(lk);
			}
			if (task)
			{
				availableNum--;
				task();
				availableNum++;
			}
		}
	}
};
