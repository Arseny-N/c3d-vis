# Visualizations of video classification networks

An adaptation of lucid to visualizing features for the I3D video classification network.
The video classification network pretrained on the kinetics dataset [weights source](https://github.com/deepmind/kinetics-i3d).

At the time of the development of this code the n-dimensional real inverse furrier transform was not differentiable in TensorFlow. In order to bypass this a function turning the spectrum of a real signal to a hermitian array was developed, this allowed to make use of the complex inverse transform, which was differentiable, the source code could be found [here](https://github.com/Arseny-N/c3d-vis/blob/9c943fd46646ba9ca67c30f00cfa104cb64e7c1f/lucid_video.py#L309).

# Results

Click on the images to see a higher resolution video.

## CPPN parametrizations

[![video](https://raw.githubusercontent.com/Arseny-N/c3d-vis/master/images/cppn.gif)](https://www.youtube.com/watch?v=mfwh_MVclHc)


## Alpha parametrizations

[![video](https://raw.githubusercontent.com/Arseny-N/c3d-vis/master/images/alpha.gif)](https://www.youtube.com/watch?v=kM8-8piGXfA)

## Shared image parametrizatoins 

[![video](https://raw.githubusercontent.com/Arseny-N/c3d-vis/master/images/shared-image-parametrization.gif)](https://www.youtube.com/watch?v=-ZLReIKM6RU)







## Neuron Visualizations

[![video](https://raw.githubusercontent.com/Arseny-N/c3d-vis/master/images/neuron.gif)](https://youtu.be/UD0vPGpnWbk)

## Channel Visualizations

### Upper layers

[![video](https://raw.githubusercontent.com/Arseny-N/c3d-vis/master/images/upper_layers.gif)](https://youtu.be/hSyV6KqzVk4)

### Middle layers

[![video0](https://raw.githubusercontent.com/Arseny-N/c3d-vis/master/images/middle_layers_0.gif)](https://youtu.be/gqOiKc8V0Io)
[![video1](https://raw.githubusercontent.com/Arseny-N/c3d-vis/master/images/middle_layers_1.gif)](https://youtu.be/TKAXHMWmJDU)
[![video2](https://raw.githubusercontent.com/Arseny-N/c3d-vis/master/images/middle_layers_2.gif)](https://youtu.be/ecemlLphnsc)
[![video3](https://raw.githubusercontent.com/Arseny-N/c3d-vis/master/images/middle_layers_3.gif)](https://youtu.be/gh4YevZwdxo)

### Lower layers

[![video](https://raw.githubusercontent.com/Arseny-N/c3d-vis/master/images/lower_layers.gif)](https://youtu.be/W4zUmejOVlA)

