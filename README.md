# MultiScaleDL
Multi-scale Deep Learning architecture for horizontal velocity estimation on the solar surface

# Abstract
aaaaa

# Dependencies
The network (network_definition.py) and the sample code (sample_draft3.py) depend on following packages
 - Keras (tested with ver2.3.1)
 - Tensorflow (tested with ver1.15.0)

# Sample data
To demonstrate our network, we provide a set of MHD simulation data (Riethmüller et al. 2014), which includes followings:
 - spatial distributions of vertical velocity and temperature of three consective frames
 - spatial distribution of corresponding horizontal velocity (y-component) to be estimated by the network.
The data files are MEMMAP files defined in the numpy package.

