import torch
import torch.nn as nn
from tqdm import tqdm
from model import KQnet
from data import train_loader,test_loader
import torch.optim as optim

epoches = 100
learning_rate = 0.0001
train_number = len(train_loader)
test_number = len(test_loader)


def main():
    device = torch.device('cuda')
    print(device)

    best_loss = 1e10

    net = KQnet()
    net.to(device)

    loss_function = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    for epoch in range(epoches):
        net.train()
        train_bar = tqdm(train_loader)
        running_loss = 0.0
        for step,data in enumerate(train_bar):
            feature,label = data
            feature = feature.to(device)
            label = label.to(device)


            output=net(feature)
            output = torch.squeeze(output)  #维度不对计算错误很大
            loss = loss_function(output,label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss = running_loss+loss.item()

        if running_loss<best_loss:
            best_loss = running_loss
            torch.save(net.state_dict(),'1.pth')

        print(epoch,running_loss)

        with open('1.txt','+a') as fw:
            fw.write(str(epoch)+' '+str(running_loss)+' '+'\n')


main()

