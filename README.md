# MultiScaleDL
Multi-scale Deep Learning architecture for horizontal velocity estimation on the solar surface is provided.
This network requires three consecutive images of vertical velocity and temperature with the time cadence of 35 seconds as input.
The detail of the network and the evaluation are described in our paper (Ishikawa et al. 2021, accepted to A&A).

# Abstract
The dynamics in the photosphere is governed by the multi-scale turbulent convection termed as granulation and supergranulation.
It is important to derive three-dimensional velocity vectors to understand the nature of the turbulent convection
and to evaluate the vertical Poynting flux toward the upper atmosphere.
The line-of-sight component of the velocity can be obtained by observing the Doppler shifts.
However, it is difficult to obtain the velocity component perpendicular to the line-of-sight,
which corresponds to the horizontal velocity in disk center observations.
We developed a convolutional neural network model with a multi-scale deep learning architecture.
The method consists of multiple convolutional kernels with various sizes of the receptive fields,
and it performs convolution for spatial and temporal axes.
The network is trained with data from three different numerical simulations of turbulent convection.
Furthermore, we introduced a novel coherence spectrum to assess the horizontal velocity fields that were derived at each spatial scale.
For more detail, see the paper Ishikawa et al. (2021).

# Dependencies
The network (network_definition.py) and the sample code (sample_calculation.py) depend on following packages
 - Keras (tested with ver2.3.1)
 - Tensorflow (tested with ver1.15.0)

# Sample data
The optimized model provided here is trained with the MURaM data.
To demonstrate our network, we provide a set of MURaM data (Riethmüller et al. 2014), including
 - spatial distributions of vertical velocity and temperature of three consecutive frames
 - spatial distribution of corresponding horizontal velocity (y-component) to be estimated by the network.

The data files are provided with MEMMAP format defined in the numpy package.

