import torch.nn as nn
import torch.utils.data.distributed
from models.densenet import densenet201
from torchvision import transforms
from torchvision import datasets
import os
import models
import datagen


def densenet(cls=28):
    model_conv = densenet201(pretrained=False)
    model_conv.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model_conv.classifier.in_features
    model_conv.classifier = nn.Linear(num_ftrs, cls)
    return model_conv


def pnasnet(cls=61):
    # model_conv = models.dense_nonorm.densenet201(pretrained=False)
    # num_ftrs = model_conv.classifier.in_features
    # model_conv.classifier = nn.Linear(num_ftrs, cls)
    model_conv = models.pnasnet.pnasnet5large(cls, None)
    num_ftrs = model_conv.last_linear.in_features
    model_conv.last_linear = nn.Linear(num_ftrs, cls)
    return model_conv


if __name__ == '__main__':
    model = densenet().cuda()
    # print(model)
    checkpoint = torch.load('checkpoint/try_3_dense201temp.t7')
    model.load_state_dict(checkpoint['net'])
    # directory = '/root/palm/DATA/plant/ai_challenger_pdr2018_testA_20180905/AgriculturalDisease_testA/'
    # directory = '/home/palm/PycharmProjects/DATA/ai_challenger_pdr2018_testA_20180905/AgriculturalDisease_testA/'
    directory = '/media/palm/data/Human Protein Atlas/test'
    directory = '/root/palm/DATA/HPAIC/train'
    out = []
    c = 0
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    model.eval()
    correct = 0
    total = 0
    files = sorted(os.listdir(directory))
    test_dataset = datagen.TestGen(directory,
                                   28,
                                   (224, 224),
                                   )
    val_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1,
        pin_memory=False)
    with torch.no_grad():
        with open('prd/1_dense201.csv', 'w') as wr:
            wr.write('Id,Predicted\n')
            for batch_idx, inputs in enumerate(val_loader):
                inputs = inputs.to('cuda')
                outputs = model(inputs)
                y = torch.ones(outputs.shape).cuda()
                print(outputs.cpu().detach().numpy())
                break
                predicted = torch.where(outputs > 0.5, outputs, y).cpu().detach().numpy()[0]

                imname = files[batch_idx * 4].split('_')[0]
                wr.write(f'{imname},')
                for i in range(len(predicted)):
                    if predicted[i] > 0:
                        wr.write(f'{i} ')
                wr.write('\n')
                print(batch_idx, end='\r')
