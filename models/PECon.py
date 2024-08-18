import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

#W 
from .penet_classifier import PENetClassifier

#W 
class CLIP(nn.Module):
    def __init__(self, img_model, ehr_model):
        super(CLIP,self).__init__()
        self.visual = img_model
        self.text = ehr_model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features

    def forward(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        return image_features, text_features, self.logit_scale.exp()

#W
class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss=local_loss
        self.gather_with_grad=gather_with_grad
        self.cache_labels=cache_labels
        self.rank=rank
        self.world_size=world_size
        self.use_horovod=use_horovod

        self.prev_num_logits=0
        self.labels={}

    def forward(self,image_features,text_features,logits_scale):
        device=image_features.device
        if self.world_size>1:
            all_image_features,all_text_features=image_features,text_features
            if self.local_loss:
                logits_per_image=logits_scale*image_features@all_text_features.T
                logits_per_text=logits_scale*text_features@all_image_features.T
            else:
                logits_per_image=logits_scale*all_image_features@all_text_features.T
                logits_per_text=logits_per_image.T
        else:
            logits_per_image=logits_scale*image_features@text_features.T
            logits_per_text=logits_scale*text_features@image_features.T

        num_logits=logits_per_image.shape[0]
        if self.prev_num_logits!=num_logits or device not in self.labels:
            labels=torch.arange(num_logits,device=device,dtype=torch.long)
            if self.world_size>1 and self.local_loss:
                labels=labels+num_logits*self.rank
            if self.cache_labels:
                self.labels[device]=labels
                self.prev_num_logits=num_logits
        else:
            labels=self.labels[device]

        total_loss=(
            F.cross_entropy(logits_per_image,labels)+
            F.cross_entropy(logits_per_text,labels)
            )/2
        return total_loss

#W 
class IMG_MLP(nn.Module):
    def __init__(self,**kwargs):
        super(IMG_MLP,self).__init__()
        self.phase=kwargs.get('phase')
        self.img_ext=PENetClassifier()
        self.fc1=nn.Linear(2048,512)
        self.fc2=nn.Linear(512,256)
        self.dropout1=nn.Dropout(0.5)
        self.fc3=nn.Linear(256,128)
        
        nn.init.kaiming_normal_(self.fc1.weight,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight,mode='fan_in',nonlinearity='relu')

        self.load_pretrained()

    def load_pretrained(self):
        if self.phase=='train':
            trained_dict=torch.load('/mntcephfs/lab_data/wangcm/wangzhipeng/ckpt/penet.pth.tar')['model_state']
            model_dict=self.img_ext.state_dict()
            trained_dict={k[len('module.'):]:v for k,v in trained_dict.items()}
            trained_dict={k:v for k,v in trained_dict.items() if k in model_dict}
            model_dict.update(trained_dict)
            self.img_ext.load_state_dict(model_dict)
            print('load_pretrained_IMG_MLP_img_ext')
            
    def train(self,mode=True):
        if mode:
            for module in self.children():
                module.train(mode)
            self.img_ext.eval()
            for param in self.img_ext.parameters():
                param.requires_grad=False
        else:
            for module in self.children():
                module.train(mode)

    def forward(self,data):
        data,_=self.img_ext(data)
        data=nn.AdaptiveAvgPool3d(1)(data)
        data=data.view(data.size(0),-1)
        z1=F.relu(self.fc1(data))
        z1=self.dropout1(z1)
        z1=F.relu(self.fc2(z1))
        z1=(self.fc3(z1))
        
        return z1
    
#W
class EHR_MLP(nn.Module):
    def __init__(self):
        super(EHR_MLP,self).__init__()
        self.fc1=nn.Linear(49,128)
        self.fc2=nn.Linear(128,128)
        self.dropout1=nn.Dropout(0.4)
        
        nn.init.kaiming_normal_(self.fc1.weight,mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight,mode='fan_in',nonlinearity='relu')

    def forward(self,data):
        z1=F.relu(self.fc1(data))
        z1=self.dropout1(z1)
        z1=(self.fc2(z1))
        
        return z1
    
#W
class PECon(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.phase=kwargs.get('phase')
        self.img_mlp=IMG_MLP(phase=self.phase)
        self.tab_mlp=EHR_MLP()
        self.clip=CLIP(self.img_mlp,self.tab_mlp)
        self.clip_loss=ClipLoss()
        
    def forward(self,i,t):
        i,t,scale=self.clip(i,t)
        loss=self.clip_loss(i,t,scale)

        return loss
    
    def args_dict(self):
        model_args={}
        return model_args

#W
class IMG_classifier(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.phase=kwargs.get('phase')
        self.img_model = IMG_MLP(phase=self.phase)
        self.classifier = nn.Linear(128,1)
    
        nn.init.kaiming_normal_(self.classifier.weight,mode='fan_in',nonlinearity='relu')
        
        self.load_pretrained()

    def load_pretrained(self):
        if self.phase=='train':
            trained_dict=torch.load('/mntcephfs/lab_data/wangcm/wangzhipeng/penet_new_train_log/PECon_1-50/best.pth.tar')['model_state']
            model_dict=self.img_model.state_dict()
            trained_dict={k[len('module.'):]:v for k,v in trained_dict.items()}
            trained_dict={k:v for k,v in trained_dict.items() if k in model_dict}
            model_dict.update(trained_dict)
            self.img_model.load_state_dict(model_dict)
            print('load_pretrained_IMG_classifier_img_model')

    def forward(self, data):
        z1 = self.img_model(data)
        z1 = self.classifier(z1)
        
        return z1
    
    def args_dict(self):
        model_args={}
        return model_args
    
#W
class EHR_classifier(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.phase=kwargs.get('phase')
        self.ehr_model = EHR_MLP()
        self.classifier = nn.Linear(128,1)

        self.load_pretrained()

    def load_pretrained(self):
        if self.phase=='train':
            trained_dict=torch.load('/mntcephfs/lab_data/wangcm/wangzhipeng/penet_new_train_log/PECon_1-50/best.pth.tar')['model_state']
            model_dict=self.ehr_model.state_dict()
            trained_dict={k[len('module.'):]:v for k,v in trained_dict.items()}
            trained_dict={k:v for k,v in trained_dict.items() if k in model_dict}
            model_dict.update(trained_dict)
            self.ehr_model.load_state_dict(model_dict)
            print('load_pretrained_EHR_classifier_ehr_model')

    def forward(self, data):
        z1 = self.ehr_model(data)
        z1 = self.classifier(z1)
        
        return z1
    
    def args_dict(self):
        model_args={}
        return model_args