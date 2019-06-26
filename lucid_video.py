
from lucid.modelzoo.vision_base import Model, _layers_from_list_of_dicts


#
# models
#

class I3D(Model):
    
    model_path = './models/i3d.pb'
  # labels_path = 'gs://modelzoo/labels/ImageNet_standard.txt'
  # synsets_path = 'gs://modelzoo/labels/ImageNet_standard_synsets.txt'
    dataset = 'Kinetiks'
    image_shape = [None, None, None, 3]
    image_rank = 4
    image_value_range = (0, 1)
    input_name = 'input'

I3D.layers = _layers_from_list_of_dicts(I3D, [
        {'tags': ['conv'], 'name': 'inceptioni3d/Conv3d_1a_7x7/Relu', 'depth': 64},
        {'tags': ['conv'], 'name': 'inceptioni3d/Conv3d_2b_1x1/Relu', 'depth': 64},
        {'tags': ['conv'], 'name': 'inceptioni3d/Conv3d_2c_3x3/Relu', 'depth': 192},
        {'tags': ['conv'], 'name': 'inceptioni3d/Mixed_3b/concat', 'depth': 256},
        {'tags': ['conv'], 'name': 'inceptioni3d/Mixed_3c/concat', 'depth': 480},
        {'tags': ['conv'], 'name': 'inceptioni3d/Mixed_4b/concat', 'depth': 512},
        {'tags': ['conv'], 'name': 'inceptioni3d/Mixed_4c/concat', 'depth': 512},
        {'tags': ['conv'], 'name': 'inceptioni3d/Mixed_4d/concat', 'depth': 512},
        {'tags': ['conv'], 'name': 'inceptioni3d/Mixed_4e/concat', 'depth': 528},
        {'tags': ['conv'], 'name': 'inceptioni3d/Mixed_4f/concat', 'depth': 832},
        {'tags': ['conv'], 'name': 'inceptioni3d/Mixed_5b/concat', 'depth': 832},
        {'tags': ['conv'], 'name': 'inceptioni3d/Mixed_5c/concat', 'depth': 1024},
        {'tags': ['conv'],
        'name': 'inceptioni3d/Logits/SpatialSqueeze',
        'depth': 400}
])


#
# render
#

from common import showarray, visstd

from tqdm import tqdm_notebook  as tqdm

from lucid.optvis.render import make_vis_T, make_print_objective_func

def render_vis(model, objective_f, param_f=None, optimizer=None,
               transforms=None, thresholds=(512,), print_objectives=None,
               verbose=True, relu_gradient_override=True, use_fixed_seed=False):
    with tf.Graph().as_default() as graph, tf.Session() as sess:

        if use_fixed_seed:  # does not mean results are reproducible, see Args doc
            tf.set_random_seed(0)

        T = make_vis_T(model, objective_f, param_f, optimizer, transforms,
                   relu_gradient_override)
        print_objective_func = make_print_objective_func(print_objectives, T)
        loss, vis_op, t_image = T("loss"), T("vis_op"), T("input")
        tf.global_variables_initializer().run()

        images = []
        try:
            for i in tqdm(range(max(thresholds)+1)):
                loss_, _ = sess.run([loss, vis_op])
                if i in thresholds:
                    vis = t_image.eval()
                    images.append(vis)
                    if verbose:
                        # print(i, loss_)
                        print_objective_func(sess)
                        showarray(visstd(vis))
        except KeyboardInterrupt:
            # log.warning("Interrupted optimization at step {:d}.".format(i+1))
            vis = t_image.eval()
            showarray(visstd(vis))

    return images

#
# objectives
#

from lucid.optvis.objectives import wrap_objective

@wrap_objective
def neuron(layer_name, channel_n, x=None, y=None, t=None, batch=None):
    """Visualize a single neuron of a single channel.

    Defaults to the center neuron. When width and height are even numbers, we
    choose the neuron in the bottom right of the center 2x2 neurons.

    Odd width & height:               Even width & height:

    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   | X |   |                     |   |   |   |   |
    +---+---+---+                     +---+---+---+---+
    |   |   |   |                     |   |   | X |   |
    +---+---+---+                     +---+---+---+---+
                                    |   |   |   |   |
                                    +---+---+---+---+
    """
    def inner(T):
        layer = T(layer_name)
        shape = tf.shape(layer)
        t_ = shape[1] // 2 if t is None else t
        x_ = shape[2] // 2 if x is None else x
        y_ = shape[3] // 2 if y is None else y
        
        if batch is None:
            return layer[:, t_, x_, y_, channel_n]
        else:
            return layer[batch, t_, x_, y_, channel_n]
    return inner

@wrap_objective
def frame(layer_name, channel_n, t=None, batch=None):
  
    def inner(T):
        layer = T(layer_name)
        shape = tf.shape(layer)
        t_ = shape[1] // 2 if t is None else t
        
        if batch is None:
            return layer[:, t_, :, :, channel_n]
        else:
            return layer[batch, t_, :, :, channel_n]
    return inner


#
# params
#

from lucid.misc.io import show

import lucid.optvis.render as render
from lucid.optvis.param.color import to_valid_rgb
# from lucid.optvis.param.spatial import pixel_image, fft_image



def rfft3d_freqs(t, h, w):

    fy = np.fft.fftfreq(h)[None, :, None]
    ft = np.fft.fftfreq(t)[:, None, None]
    # when we have an odd input dimension we need to keep one additional
    # frequency and later cut off 1 pixel
    if w % 2 == 1:
        fx = np.fft.fftfreq(w)[None, None, : w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w)[None, None, : w // 2 + 1]
    return np.sqrt(ft * ft + fx * fx + fy * fy)


def to_herm_nd_tf(s):
    """
    Turn the spectrum of a real signal `s' to a hermitian array
    """
    cat = lambda a, b, axis: tf.concat([a, b], axis)
    
    flip_row = lambda row, axis:\
        cat(tf_take(row, slice(0, 1), axis), \
               sub_and_filp(row,  axis), axis)  
    
    def tf_take(x, ixs, axis):
        xs = [slice(None)]*len(x.shape)
        xs[axis] = ixs
        return x.__getitem__(tuple(xs))
    
    def sub_and_filp(s, axis):    
        x = tf_take(s, slice(1, None), axis)
        return tf.reverse(x, axis=[axis])
    
    skip_last = int(s.shape[-1]) % 2 == 1 
    s1 = tf.reverse(s[..., slice(1, int(s.shape[-1]) - skip_last)], axis=[-1])
    
    rows = []
    for dim in range(2, len(s.shape)+1):
        
        row = tf_take(s1, slice(0, 1), -dim)
        
        s1 = sub_and_filp(s1, -dim)
        
        rows = [ flip_row(row, -dim) for row in rows ] 
        rows.append(row)
        
    
    rows = reversed(rows)
    dims = reversed(range(2, len(s.shape)+1))
    r = s1
    
    for dim, row in zip(dims, rows):
        r = cat(row, r, -dim)
        
    return cat(s, tf.conj(r), -1)

def fft_video(shape, sd=None, decay_power=1):

    sd = 0.01
    batch, t, h, w, ch = shape
    freqs = rfft3d_freqs(t, h, w)
    init_val_size = (2, ch) + freqs.shape
    
    images = []
    for _ in range(batch):
        init_val = np.random.normal(size=init_val_size, scale=sd).astype(np.float32)
        spectrum_real_imag_t = tf.Variable(init_val)
        spectrum_t = tf.complex(spectrum_real_imag_t[0], spectrum_real_imag_t[1])

        scale = 1.0 / np.maximum(freqs, 1.0 / max(w, h)) ** decay_power
        scale *= np.sqrt(w * h)
        scaled_spectrum_t = scale * spectrum_t
        
        
        # No backwards for ifft3d, so add the complex conjugate to the spectrum.
        # Another way would be to do ifft3d is by composing several ifft1d's
        scaled_spectrum_t = to_herm_nd_tf(scaled_spectrum_t)
 
        image_t = tf.spectral.ifft3d(scaled_spectrum_t)
        image_t = tf.transpose(image_t, (1, 2, 3, 0))
        image_t = tf.real(image_t)

        image_t = image_t[:t, :h, :w, :ch]
        images.append(image_t)

    batched_image_t = tf.stack(images) / 4.0  # TODO: is that a magic constant?
    return batched_image_t

def uniform_video(t, w, h=None, batch=None, offset=100.):
    h = h or w
    batch = batch or 1
    channels = 3
    shape = [batch, t, w, h, channels]

    x = tf.Variable(np.random.uniform(size=shape).astype(np.float32) + 100.)
    return tf.identity(x)

def video(t, w, h=None, batch=None, sd=None, decorrelate=True, fft=True, alpha=False):
    h = h or w
    batch = batch or 1
    channels = 4 if alpha else 3
    shape = [batch, t, w, h, channels]

    param_f = fft_video if fft else uniform_video
    t = param_f(shape, sd=sd)
    rgb = to_valid_rgb(t[..., :3], decorrelate=decorrelate, sigmoid=True)
    if alpha:
        a = tf.nn.sigmoid(t[..., 3:])
        return tf.concat([rgb, a], -1)
    return rgb


#
# transforms
#

import tensorflow as tf

def pad_with_t(w, t, mode="REFLECT", constant_value=0.5):
    def inner(t_image):
        if constant_value == "uniform":
            constant_value_ = tf.random_uniform([], 0, 1)
        else:
            constant_value_ = constant_value
        return tf.pad(
            t_image,
            [(0, 0), (t, t), (w, w), (w, w), (0, 0)],
            mode=mode,
            constant_values=constant_value_,
        )

    return inner


def compose_video(transforms):
    def inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return inner


def wrap_transform(transform, levels):

    def fn(video, its, transform):
        video = tf.convert_to_tensor(video, preferred_dtype=tf.float32)
        return tf.map_fn(transform, video, 
            parallel_iterations=its, back_prop=True)

    if levels == 2:
        return lambda video: fn(video, 10, transform)
    return lambda video: fn(video, 5, lambda image: fn(image, 2, transform))





from lucid.optvis.transform import  pad






from tensorflow.python.ops.random_ops import *
import sys



def jitter_image(t_image, seed, d):
    t_image = tf.convert_to_tensor(t_image, preferred_dtype=tf.float32)
    t_shp = tf.shape(t_image)
    crop_shape = tf.concat([t_shp[:-3], t_shp[-3:-1] - d, t_shp[-1:]], 0)
    if seed:
        tf.set_random_seed(seed)
    crop = tf.random_crop(t_image, crop_shape, seed=seed)
    shp = t_image.get_shape().as_list()
    mid_shp_changed = [
        shp[-3] - d if shp[-3] is not None else None,
        shp[-2] - d if shp[-3] is not None else None,
    ]
    crop.set_shape(shp[:-3] + mid_shp_changed + shp[-1:])
    return crop

import functools as ft
import time


def jitter(d):
    def inner(t_video):
        # print(t_video.shape)
        seed = int(time.time())
        seed = None
        transform = ft.partial(jitter_image, seed=seed, d=d)
        return wrap_transform(transform, levels=3)(t_video)
    return inner

def _angle2rads(angle, units):
    angle = tf.cast(angle, "float32")
    if units.lower() == "degrees":
        angle = 3.14 * angle / 180.
    elif units.lower() in ["radians", "rads", "rad"]:
        angle = angle
    return angle

def _rand_select(xs, seed=None):
    xs_list = list(xs)
    if seed:
        tf.set_random_seed(seed)
    rand_n = tf.random_uniform((), 0, len(xs_list), "int32", seed=seed)
    return tf.constant(xs_list)[rand_n]


def random_scale_image(t, seed, scales):
    t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
    scale = _rand_select(scales, seed=seed)
    shp = tf.shape(t)
    scale_shape = tf.cast(scale * tf.cast(shp[-3:-1], "float32"), "int32")
    return tf.image.resize_bilinear(t, scale_shape)

def random_scale(scales):
    def inner(t):
        seed = int(time.time())
        seed = None
        transform = ft.partial(random_scale_image, seed=seed, scales=scales)
        return wrap_transform(transform, levels=2)(t)
    return inner



def random_rotate_image(t, seed, angles, units="degrees"):
    t = tf.convert_to_tensor(t, preferred_dtype=tf.float32)
    angle = _rand_select(angles, seed=seed)
    angle = _angle2rads(angle, units)
    return tf.contrib.image.rotate(t, angle)

def random_rotate(angles, units="degrees", seed = None):
    def inner(t):
        transform = ft.partial(random_rotate_image, seed=seed, angles=angles, units=units)
        return wrap_transform(transform, levels=3)(t)
    return inner


def pad_image(t_image, w, mode="REFLECT", constant_value=0.5):
    if constant_value == "uniform":
        constant_value_ = tf.random_uniform([], 0, 1)
    else:
        constant_value_ = constant_value
    return tf.pad(
        t_image,
        [(0, 0), (w, w), (w, w), (0, 0)],
        mode=mode,
        constant_values=constant_value_,
    )

def pad(w, mode="REFLECT", constant_value=0.5, seed = None):
    def inner(t):
        transform = ft.partial(pad_image, w=w, mode=mode, constant_value=constant_value)
        return wrap_transform(transform, levels=2)(t)
    return inner


standard_transforms = [
    pad(12, mode="constant", constant_value=.5),
    jitter(8),
    random_scale([1 + (i - 5) / 50. for i in range(11)]),
    random_rotate(list(range(-10, 11)) + 5 * [0]),
    jitter(4),
]




def hang_video(subclip_len = 10):
    def inner(video):
        
        clip_len = video.shape[1]
        
        b = tf.random_uniform([1], minval=0, maxval=clip_len-subclip_len, dtype=tf.int32)[0]
        e = b+subclip_len
        
        return tf.concat([
            video[:, :b, :, :, :], 
            tf.tile(video[:, b:b+1, :, :, :], [1, e-b, 1,1,1]),
            video[:, e:, :, :, :]], axis=1) 
    return inner




def lap_normalize(img, scale_n=4, k0=[1,4,6,4,1]):
    '''Perform the Laplacian pyramid normalization.'''
    
    k0 = np.float32(k0)
    k1 = k0[None, None]
    k1 = (k1.transpose(0, 2, 1) * k1 * k1.transpose(2, 0, 1))
    k5x5x5 = k1[:,:,:,None,None] * np.eye(3, dtype=np.float32)

    def lap_split(img):
        '''Split the image into lo and hi frequency components'''
        with tf.name_scope('split'):

            lo = tf.nn.conv3d(img, k5x5x5, [1,2,2,2,1], 'SAME')
            lo2 = tf.nn.conv3d_transpose(lo, k5x5x5*4, tf.shape(img), [1,2,2,2,1])
            hi = img-lo2
        return lo, hi

    def lap_split_n(img, n):
        '''Build Laplacian pyramid with n splits'''
        levels = []
        for i in range(n):
            img, hi = lap_split(img)
            levels.append(hi)
        levels.append(img)
        return levels[::-1]

    def lap_merge(levels):
        '''Merge Laplacian pyramid'''
        img = levels[0]
        for hi in levels[1:]:
            with tf.name_scope('merge'):
                img = tf.nn.conv3d_transpose(img, k5x5x5*4, tf.shape(hi), [1,2,2,2,1]) + hi
        return img

    def normalize_std(img, eps=1e-10):
        '''Normalize image by making its standard deviation = 1.0'''
        with tf.name_scope('normalize'):
            std = tf.sqrt(tf.reduce_mean(tf.square(img)))
            return img/tf.maximum(std, eps)

    tlevels = lap_split_n(img, scale_n)
    tlevels = list(map(normalize_std, tlevels))
    out = lap_merge(tlevels)
    return out


def normalize_gradient_by_lap(scale_n=4):
    import uuid 
    
    op_name = "NormalizeGradByLap_" + str(uuid.uuid4())

    @tf.RegisterGradient(op_name)
    def _NormalizeGradByLap(op, grad):
        return lap_normalize(grad, scale_n=scale_n)

    def inner(x):
        with x.graph.gradient_override_map({"Identity": op_name}):
            x = tf.identity(x)
        return x

    return inner



def normalize_gradient_by_std():
    import uuid 
    
    op_name = "NormalizeGradByStd_" + str(uuid.uuid4())

    @tf.RegisterGradient(op_name)
    def _NormalizeGradByStd(op, grad):
        std = tf.math.reduce_std(grad)
        return grad / std +1e-8   

    def inner(x):
        with x.graph.gradient_override_map({"Identity": op_name}):
            x = tf.identity(x)
        return x

    return inner
