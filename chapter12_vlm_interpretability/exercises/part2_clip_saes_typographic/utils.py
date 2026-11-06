"""Rendering only; feature extraction and sparse coding remain visible in the master."""
import numpy as np
from PIL import Image, ImageDraw

NAMES=('circle','square','triangle')
COLOURS=((240,30,30),(30,220,30),(30,30,240))

def render_card(object_id,word_id=None,show_object=True):
 image=Image.new('RGB',(64,64),(0,0,0));draw=ImageDraw.Draw(image)
 if show_object:
  colour=COLOURS[object_id]
  if object_id==0:draw.ellipse((12,2,52,29),fill=colour)
  elif object_id==1:draw.rectangle((12,2,52,29),fill=colour)
  else:draw.polygon([(32,2),(12,29),(52,29)],fill=colour)
 if word_id is not None:draw.text((2,42),NAMES[word_id],fill='white')
 return np.asarray(image,dtype=float)/255

def word_templates():
 # Last template is the blank label region, providing an explicit no-text channel.
 return np.stack([render_card(0,j,False)[32:] for j in range(3)]+[render_card(0,None,False)[32:]])
