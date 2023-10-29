import torch.nn as nn
import math
from torchvision import models

class CNN_model(nn.Module):
    def __init__(self, input_size, channels):
        super(CNN_model,self).__init__()
        
        self.name = "CNN Classifier"
        filter_size = 5
        
        
        self.conv1 = nn.Conv2d(channels,16,kernel_size = (filter_size,filter_size), stride=1)
        self.batch_norm1 = nn.BatchNorm1d(16)
        self.max_pool = nn.MaxPool2d(3, stride=2)
        

        self.conv2 = nn.Conv2d(16,32, kernel_size = (filter_size,filter_size), stride = 1)
        self.batch_norm2 = nn.BatchNorm1d(32)
        

        self.conv3 = nn.Conv2d(32,64, kernel_size = (filter_size,filter_size), stride = 1)
        self.batch_norm3 = nn.BatchNorm1d(64)
        
        self.conv4 = nn.Conv2d(64,32, kernel_size = (filter_size,filter_size), stride = 1)
        self.batch_norm3 = nn.BatchNorm1d(64)
        
        self.conv5 = nn.Conv2d(32,16, kernel_size = (filter_size,filter_size), stride = 1)
        self.batch_norm3 = nn.BatchNorm1d(64)
    
       
        self.flatten_size = 64
        self.flatten = nn.Flatten(start_dim=1)
        
        self.Linear1 = nn.Linear(self.flatten_size, input_size)
        self.batch_norm_linear = nn.BatchNorm1d(input_size)
        # self.a = nn.Linear()
        self.Linear2 = nn.Linear(input_size,4)
        
        self.Softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        
        
    def forward(self,x):
        # x= x.view(x.shape[0],1,x.shape[1])
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.max_pool(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.max_pool(out)   

        out = self.conv3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.max_pool(out) 
        
        out = self.conv4(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.max_pool(out) 
        
        out = self.conv5(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.max_pool(out) 

        out = self.flatten(out)
        
        out = self.Linear1(out)  
        out = self.Linear2(out)
        out = self.Softmax(out)
        
        return out
    

        
class train_transfer_learning_model(nn.Module):    
    
    def __init__(self,model_name):
        super(train_transfer_learning_model,self).__init__()
            
        if(model_name =="ResNet"):
            print("Traning ResNet")
            self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self.name  = "ResNet"

            for param in self.model.parameters():
                param.requires_grad = False                     #Freezing all the layers and changing only the below layers

            self.model.fc = nn.Linear(2048,4)
            self.model.aux_logits = False
        elif(model_name == "GoogleNet"):
            print("Traning GoogleNet")
            self.model =  models.googlenet()
            self.name  = "GoogleNet"

            for param in self.model.parameters():
                param.requires_grad = False                     #Freezing all the layers and changing only the below layers

            self.model.fc = nn.Linear(1024,4)
            self.model.aux_logits = False

        else:
            print("Traning MobileNet")
            self.model =  models.mobilenet_v3_large()
            self.name  = "MobileNet"
    
            for param in self.model.parameters():
                param.requires_grad = False                     #Freezing all the layers and changing only the below layers

            self.model.classifier = nn.Linear(self.model.classifier[0].in_features, 4)
    
    def forward(self,x):
        
        out = self.model(x) 
        return out
                