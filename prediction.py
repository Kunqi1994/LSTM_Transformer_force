import torch
from tqdm import tqdm
from model import KQnet
from data import test_loader,force_max, force_min


def eval():
    device = torch.device('cuda')
    print(device)

    net = KQnet()
    net.to(device)
    net.eval()

    force_span = force_max - force_min
    force_span = force_span.item()

    net.load_state_dict(torch.load('1.pth'))

    with torch.no_grad():
        test_bar = tqdm(test_loader)
        for data in test_bar:
            feature, label = data
            feature = feature.to(device)
            label = label.to(device)
            label = label.item()
            output = net(feature)
            output = torch.squeeze(output).item()

            output = output*force_span + force_min
            label = label*force_span + force_min

            print(output,label)
            with open('2.txt', '+a') as fw:
                fw.write(str(output) + str(label) + '\n')

eval()