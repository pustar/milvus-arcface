from milvus import Milvus, IndexType, MetricType, Status
import numpy as np

m = Milvus(host='IP', port='19530')

# 创建collection
param = {
    'collection_name':'face',
    'dimension':256,
    'index_file_size':256,
    'metric_type':MetricType.IP #相似度计算方式使用內积
}
print(m.create_collection(param))

num = 200000
step = 5000
now = 0

def GetBatch(data):
    global now

    ids = np.zeros(step,dtype=np.int32)
    vects = np.zeros((step,256),dtype=np.float32)

    for i in range(step):
        tmp = data[i+now].split("|")
        ids[i] = int(tmp[0])
        for u in range(256):
            vects[i][u] = float(tmp[u+1])
    now += step
    return ids.tolist() , vects.tolist()

# 将所以人脸向量插入Milvus
data = open("G:\\feature\\res.txt").readlines()
for i in range(int(num / step)):
    ids , vs = GetBatch(data)
    res = m.insert(collection_name='face', records=vs, ids=ids)
    print(i)

# # 创建索引 ， 如果不创建，默认FLAT索引
# ivf_param = {'nlist': 16384}
# print(m.create_index('face', IndexType.IVF_FLAT, ivf_param))


# 查询
ids , vs = GetBatch(data)
idx = int(input("index:")) # 输入一个下标，从Batch中取出第idx个进行查询
print("id:" , ids[idx]) # 输出下标为idx的特征向量的id，这里的id就是文件名。14就是CelebA数据集中的000014.jpg
search_param = {'nprobe': 16}
res = m.search(collection_name='face', query_records=[vs[idx]], top_k=3, params=search_param)
print(res)