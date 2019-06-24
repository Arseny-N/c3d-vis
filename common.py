
from pathlib import Path

from IPython.display import clear_output, Image, display, HTML
import moviepy.editor as mpy

import sys

import numpy as np

import time

def add_paths(*ps):
    base = Path('~/c3d').expanduser()

    ret = []
    for p in ps:
        if str(base / p) not in sys.path:
            sys.path.insert(0, str(base / p))
        ret.append(base / p)

    return ret





dignify_layer = lambda layer: '-'.join(layer.split('/')[2:-1]).replace('/', '_')

def render_suffix(xs):
    return ','.join( f"{k}={xs[k]}" for k in sorted(xs.keys()) if k != 'layer')



def to_clip(img):
    if img.shape[0] == 1: img = img[0]
    img = np.uint8(np.clip(img, 0, 1)*255)
    return mpy.ImageSequenceClip(list(img), fps=10)



def save_vid(img, **suffix_data):
    base_path = Path('./videos')
    
    base_path = base_path / time.strftime("%Y-%m-%d") 
    base_path.mkdir(exist_ok=True)
    
    
    

    name = time.strftime("%H-%M-%S_") 
    if 'layer' in suffix_data:
        name += dignify_layer(suffix_data['layer']) 

    suffix = render_suffix(suffix_data)
    name += '_' + suffix 
    
    path = base_path / name
    
    clip = to_clip(img)
    
    clip.write_videofile(str(path) + '.mp4')
    # if save_npz: np.savez_compressed(path, img)
    
    return Path(str(path) + '.mp4')



def showarray(img):
    clip = to_clip(img)
    display(clip.ipython_display())
    
def showarray_saved(img, **kw):
    path = save_vid(img, **kw)
    display(mpy.ipython_display(str(path)))
    

def visstd(xs, s=0.1, per_ch=False):
    '''Normalize the image range for visualization'''
    
    if per_ch:
        return np.array([(a-a.mean())/max(a.std(), 1e-4)*s + 0.5 for a in xs ])
    a = xs
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5



def array_to_clip(img):
    if img.shape[0] == 1: img = img[0]
    img = np.uint8(np.clip(img, 0, 1)*255)
    return mpy.ImageSequenceClip(list(img), fps=10)

def clip_to_array(clip):
    return np.array(list(clip.iter_frames()))


#
# video managment
#
import ast

def guess_cast(x):
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return x

def parse_path(path):
    first, *params = path.name.rstrip('.mp4').split(',')

    time, *layer, param = first.split('_') 
    layer = '_'.join(layer)
    params = { p.split('=')[0] :guess_cast(p.split('=')[1]) for p in params + [param] }
    
    return dict(date=pd.to_datetime(path.parts[-2] + ' ' + time, format="%Y-%m-%d %H-%M-%S"), 
                layer=layer, path=path, **params)

import pandas as pd

def videos_df():
    return pd.DataFrame([ parse_path(file) 
         for dir in Path('videos/').iterdir()
            for file in dir.iterdir() ]).sort_values(by='date').reset_index()

