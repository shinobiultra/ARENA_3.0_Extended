"""Rendering/grid display plumbing for the dense-feature teaching task."""
import numpy as np
PALETTE=np.array([[.12,.12,.12],[.95,.08,.08],[.08,.9,.08],[.08,.08,.95]])

def part_image(shift=0,patch=8):
 labels=np.zeros((6,8),dtype=int)
 labels[1:5,1+shift:3+shift]=1
 labels[0,1+shift:3+shift]=2
 labels[2:4,3+shift]=3
 pixels=np.repeat(np.repeat(PALETTE[labels],patch,axis=0),patch,axis=1)
 return pixels,labels

def grid_rgb(projection,shape,low=None,high=None):
 values=projection[:,:3]
 low=values.min(axis=0) if low is None else low
 high=values.max(axis=0) if high is None else high
 return np.clip((values-low)/np.maximum(high-low,1e-12),0,1).reshape(*shape,3)
