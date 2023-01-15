import torch.utils.data as data
import os
import numpy as np
import torch
#data.Dataset:
#所有子类应该override__len__和__getitem__，前者提供了数据集的大小，后者支持整数索引，范围从0到len(self)

class LiverDataset(data.Dataset):
    #创建LiverDataset类的实例时，就是在调用init初始化
    def __init__(self,root):#root表示图片路径
        self.imgs=[]
        self.labels=[]
        numList=[400,800,1200,1600,2000,2400,2800,3200,3600,4000]
        for num in numList:
            for fileName in os.listdir(r"./"+root+"/"+str(num)+"raw"):
                if(fileName[-1]!='t'):
                    continue
                img="./"+root+"/"+str(num)+"raw/"+fileName
                self.imgs.append(img)
                self.labels.append(num//400-1)

    def __getitem__(self,index):
        img=np.loadtxt(self.imgs[index])
        # print(self.imgs[index])
        img=img.reshape(1,img.shape[0],img.shape[1])
        # print(img.shape)
        img=torch.tensor(img,dtype=torch.float32)
        label=self.labels[index]
        # print("label  "+str(label))
        return img,label#返回的是图片
    

    def __len__(self):
        return len(self.imgs)#400,list[i]有两个元素，[img,mask]