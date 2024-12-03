import numpy as np

data = np.load("/DLL/pointnet2-sem-seg-changed/save_data_output.npy")
print(data[0][10])