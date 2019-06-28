
from pathlib import Path

from IPython.display import clear_output, Image, display, HTML
import moviepy.editor as mpy

import sys

import numpy as np

import time


import contextlib 

def add_paths(*ps):
    base = Path('~/c3d').expanduser()

    ret = []
    for p in ps:
        if str(base / p) not in sys.path:
            sys.path.insert(0, str(base / p))
        ret.append(base / p)

    return ret




#
# video managment
#

BASE_PATH = Path('../notebooks/videos')

dignify_layer = lambda layer: layer.replace('/', '_')

def render_suffix(xs):
    return ','.join( f"{k}={xs[k]}" for k in sorted(xs.keys()) if k != 'layer')



def to_clip(img):
    if img.shape[0] == 1: img = img[0]
    img = np.uint8(np.clip(img, 0, 1)*255)
    return mpy.ImageSequenceClip(list(img), fps=10)



def save_vid(img, **suffix_data):
    
    assert 'name' in suffix_data, 'Should provide a name'
    
    base_path = BASE_PATH
    
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
         for dir in BASE_PATH.iterdir()
            for file in dir.iterdir() ]).sort_values(by='date').reset_index()


#
# clip making
#


def batchify(xs, batch_size):
    
    assert len(xs) > 0, "Should pass some videos"
    
    ixs = list(range(0, len(xs) + batch_size  , batch_size))
    ret = []
    for a, b in zip(ixs, ixs[1:]):
        x = xs[a:b]
        if len(x) < batch_size:
            fill = mpy.VideoClip(lambda t: np.ones((*xs[0].size, 3)), duration=xs[0].duration)
            # fill = mpy.ColorClip(xs[0].size, 'blue', duration=xs[0].duration, ismask=False).to_RGB()
            x.extend([fill]* (batch_size - len(x)))
        ret.append(x)
    return ret

import textwrap

def caption_clip(clip, text, position=('center', 5), wrap=16, fontsize=10, font="DejaVu-Sans-Bold"):
    
    assert font in mpy.TextClip.list('font')
    
    text = '\n'.join(textwrap.wrap(text, wrap))                
    text_clip = mpy.TextClip(text, fontsize=fontsize, color='white', font=font)
    text_clip = text_clip.set_position(position).set_duration(clip.duration)

    return mpy.CompositeVideoClip([clip, text_clip])
    
def display_videos(videos, videos_per_row=6, width=800, captions=None, sort_by_caption=True,save_as = None, ext='.mp4',
                    **caption_kw):    
    
    
    xs = []
    txts = []
    with contextlib.ExitStack() as stack:
        for ix, v in enumerate(videos):
            
            try:
                clip = mpy.VideoFileClip(str(v)).resize(width=width/videos_per_row) 
            except:
                print(f"Failed to open {v}")
                continue
            
            if captions is not None:
                text = captions[ix]
                txts.append(text)
                clip = caption_clip(clip, text, **caption_kw)                
                
            xs.append(clip)        
            stack.callback(clip.close)
        
        if txts and sort_by_caption:
            ixs = np.lexsort((txts,))
            xs = [ xs[i] for i in ixs ] 
        
        xs = batchify(xs, videos_per_row)
        clip = mpy.clips_array(xs)
        
        display(clip.ipython_display(loop=True))
        
        if save_as is not None:
            print('Saving as ', save_as + ext)
            if ext in {'.mp4'}:
                clip.write_videofile(save_as + ext)
            elif ext in {'.gif'}:
                clip.write_gif(save_as + ext)


