import torch.nn as nn


class KQnet(nn.Module):
    def __init__(self,input_size=8,hidden_size=32,num_layers=2,output_size=1):
        super(KQnet,self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self,x):
        out,(h_0,c_0) = self.rnn(x,None) #None表示hidden state(h_n,c_n)全部用0
        x = self.fc(out[:,-1,:])
        return x











