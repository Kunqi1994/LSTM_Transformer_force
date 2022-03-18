from torch.utils.data import DataLoader,TensorDataset
import torch
from pandas import read_csv

sequence_length = 5
batchSize = 16
filepath = 'index.csv'
file_data = read_csv(filepath)

df = file_data.apply(lambda x: (x - min(x)) / (max(x) - min(x)))

force_max = file_data['force'].max()
force_min = file_data['force'].min()

x_data = df.values

length = df.shape[0]-sequence_length
feature = torch.zeros(length,sequence_length,8)
label = torch.zeros(length)

for i in range(0,length):
    label[i] = x_data[i + sequence_length, 0]
    feature[i] = torch.from_numpy(x_data[i:i+sequence_length])


train_dataset = TensorDataset(feature[0:int(length*0.8)],label[0:int(length*0.8)])
test_dataset = TensorDataset(feature[int(length*0.8):int(length*0.9)],label[int(length*0.8):int(length*0.9)])

train_loader = DataLoader(dataset=train_dataset, batch_size=batchSize, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

