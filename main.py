import argparse
import torch
from torch import optim
from model import net
from dataset import LiverDataset
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_model(model,criterion,optimizer,dataload,num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dataset_size = len(dataload.dataset)
        # print("dataset_size  "+str(dataset_size))
        epoch_loss = 0
        step = 0 #minibatch数
        for x,y in dataload:
            optimizer.zero_grad()
            inputs=x.to(device)
            labels=y.to(device)
            outputs=model(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            step += 1
            print("%d/%d,train_loss:%0.3f" % (step, dataset_size // dataload.batch_size, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
    torch.save(model.state_dict(),'model.pth' )
    return model

def train():
    model=net().to(device)
    batch_size=args.batch_size

    criterion=nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset(root='pic')
    dataloader = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True,num_workers=0)
    train_model(model,criterion,optimizer,dataloader)

def test():
    model=net()
    model.load_state_dict(torch.load('model.pth'))
    model.eval()
    batch_size=args.batch_size
    liver_dataset=LiverDataset(root='test')
    dataloaders= DataLoader(liver_dataset,batch_size=1, shuffle=True,num_workers=0)
    pred=[]
    corr=[]
    cnt=1
    for x,label in dataloaders:
        y=model(x)
        # print(y,label)
        y=torch.argmax(y)
        corr.append(label[0])
        pred.append(y)
        print(str(label[0])+"  "+str(y)+" "+str(cnt))
        cnt+=1
    pred=np.array(pred)
    corr=np.array(corr)
    num=len(np.where(pred==corr)[0])
    print("accuracy "+str(num/pred.shape[0]))

    
if __name__ == '__main__':
    #参数解析
    parser = argparse.ArgumentParser() #创建一个ArgumentParser对象
    parser.add_argument('action', type=str, help='train or test')#添加参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--weight', type=str, help='the path of the mode weight file')
    args = parser.parse_args()
    
    if args.action == 'train':
        train()
    elif args.action == 'test':
        test()