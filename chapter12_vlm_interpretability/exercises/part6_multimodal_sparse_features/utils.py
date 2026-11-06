"""Only rendered image and displayed word plumbing; no sparse-method implementation."""
import numpy as np
from PIL import Image,ImageDraw
NAMES=('red square','green circle','blue triangle')

def render_examples():
 images=[];labels=[];words=[]
 for intensity in [.6,1.]:
  for c in range(3):
   image=Image.new('RGB',(24,24));draw=ImageDraw.Draw(image)
   colour=tuple(int(intensity*255) if i==c else 0 for i in range(3))
   if c==0:draw.rectangle((4,4,20,20),fill=colour)
   elif c==1:draw.ellipse((4,4,20,20),fill=colour)
   else:draw.polygon([(12,4),(4,20),(20,20)],fill=colour)
   images.append(np.asarray(image,dtype=float)/255);labels.append(c);words.append(NAMES[c])
 return np.stack(images),np.array(labels),words
