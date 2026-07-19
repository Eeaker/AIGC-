"""Shot-adapted conservative line cleanup trained only on A0002--A0005."""
from __future__ import annotations
from pathlib import Path
import cv2,numpy as np

COLORS=np.asarray([(255,255,255),(0,0,0),(255,0,0),(0,255,0),(0,0,255)],np.uint8)

def _features(image):
 d=np.abs(image[:,:,:3,None].astype(np.int16)-COLORS.T[None,None].astype(np.int16)).sum(2); y=d.argmin(2);one=np.eye(5,dtype=np.float32)[y].transpose(2,0,1);ink=(y!=0).astype(np.uint8);dist=np.clip(cv2.distanceTransform(1-ink,cv2.DIST_L2,5),0,32)[None]/32
 return np.concatenate([one,dist.astype(np.float32)]),ink

def cleanup(image:np.ndarray,weights:Path,threshold:float=.92)->np.ndarray:
 import torch
 from torch import nn
 from torch.nn import functional as F
 class Block(nn.Module):
  def __init__(self,a,b):super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GroupNorm(8,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1),nn.GroupNorm(8,b),nn.SiLU())
  def forward(self,x):return self.net(x)
 class UNet(nn.Module):
  def __init__(self):
   super().__init__();self.e1,self.e2,self.e3=Block(6,32),Block(32,64),Block(64,96);self.mid=Block(96,128);self.d3,self.d2,self.d1=Block(224,96),Block(160,64),Block(96,32);self.out=nn.Conv2d(32,5,1)
  def forward(self,x):
   a=self.e1(x);b=self.e2(F.avg_pool2d(a,2));c=self.e3(F.avg_pool2d(b,2));m=self.mid(F.avg_pool2d(c,2));c2=self.d3(torch.cat([F.interpolate(m,c.shape[-2:],mode='bilinear',align_corners=False),c],1));b2=self.d2(torch.cat([F.interpolate(c2,b.shape[-2:],mode='bilinear',align_corners=False),b],1));a2=self.d1(torch.cat([F.interpolate(b2,a.shape[-2:],mode='bilinear',align_corners=False),a],1));return self.out(a2)
 device='cuda' if torch.cuda.is_available() else 'cpu';model=UNet().to(device);model.load_state_dict(torch.load(weights,map_location=device,weights_only=True));model.eval();feat,ink=_features(image);h,w=ink.shape;score=np.zeros((5,h,w),np.float32);count=np.zeros((h,w),np.float32);tile,stride=512,448;ys=sorted(set(range(0,max(h-tile,0)+1,stride))|{max(h-tile,0)});xs=sorted(set(range(0,max(w-tile,0)+1,stride))|{max(w-tile,0)})
 with torch.inference_mode():
  for y in ys:
   for x in xs:
    p=model(torch.from_numpy(feat[:,y:y+tile,x:x+tile]).unsqueeze(0).to(device))[0].softmax(0).cpu().numpy();score[:,y:y+tile,x:x+tile]+=p;count[y:y+tile,x:x+tile]+=1
 score/=np.maximum(count,1)[None];near=cv2.distanceTransform(1-ink,cv2.DIST_L2,5)<=12;cls=np.where(((1-score[0])>=threshold)&near,1+score[1:].argmax(0),0);return COLORS[cls]
