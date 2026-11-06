"""Rendering plumbing for a four-patch colour-question task."""
import numpy as np

def render_grid(target_patch=0,colour=0,distractor_patch=3):
 image=np.full((32,32,3),.04)
 for patch,c in [(target_patch,colour),(distractor_patch,1-colour)]:
  row,col=divmod(patch,2);image[row*16+2:row*16+14,col*16+2:col*16+14]=[.9,.05,.05] if c==0 else [.05,.05,.9]
 return image
