# MultiScaleDL
Multi-scale Deep Learning architecture for horizontal velocity estimation on the solar surface

# Abstract
The dynamics in the solar photosphere is governed by the multi-scale turbulent convection.
It is important to derive three-dimensional velocity vectors to understand the nature of the turbulent convection.
The line-of-sight component of the velocity can be obtained by observing the Doppler shifts.
However, it is difficult to obtain the velocity component perpendicular to the line-of-sight,
which corresponds to the horizontal velocity in disk center observations.
To develop

# Dependencies
The network (network_definition.py) and the sample code (sample_draft3.py) depend on following packages
 - Keras (tested with ver2.3.1)
 - Tensorflow (tested with ver1.15.0)

# Sample data
To demonstrate our network, we provide a set of MHD simulation data (Riethmüller et al. 2014), including
 - spatial distributions of vertical velocity and temperature of three consective frames
 - spatial distribution of corresponding horizontal velocity (y-component) to be estimated by the network.

The data files are provided with MEMMAP format defined in the numpy package.

