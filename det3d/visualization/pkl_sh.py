# show_pkl.py

import pickle

path = '/home/st2000/data/work_dir/try_plot/prediction.pkl'  # path='/root/……/aus_openface.pkl'   pkl文件所在路径

f = open(path, 'rb')
data = pickle.load(f)

print(data)
print(len(data))