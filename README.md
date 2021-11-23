# MultiScaleDL
Multi-scale Deep Learning architecture for horizontal velocity estimation on the solar surface is provided.
This network requires three consective images of vertical velocity and temperature with the time cadence of 35 seconds as input.
The detail of the network and the evaluation are described in our paper (Ishikawa et al. 2021 accepted to A&A).

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

The multi-scale deep learning method successfully predicts the horizontal velocities for each convection simulation
in terms of the global-correlation-coefficient, which is often used for evaluating the prediction accuracy of the methods.
The coherence spectrum reveals the strong dependence of the correlation coefficients on the spatial scales.
Although coherence spectra are higher than 0.9 for large-scale structures,
they drastically decrease to less than 0.3 for small-scale structures
wherein the global-correlation-coefficient indicates a high value of approximately 0.95.
By comparing the results of the three convection simulations,
we determined that this decrease in the coherence spectrum occurs
around the energy injection scales, which are
characterized by the peak of the power spectra of the vertical velocities.

The accuracy for the small-scale structures is not guaranteed solely by the global-correlation-coefficient.
To improve the accuracy in small-scales, it is important to improve the loss function for enhancing the small-scale structures
and to utilize other physical quantities related to the non-linear cascade of convective eddies as input data.

# Dependencies
The network (network_definition.py) and the sample code (sample_draft3.py) depend on following packages
 - Keras (tested with ver2.3.1)
 - Tensorflow (tested with ver1.15.0)

# Sample data
The optimized model provided here is trained with the MURaM data.
To demonstrate our network, we provide a set of MURaM data (Riethmüller et al. 2014), including
 - spatial distributions of vertical velocity and temperature of three consective frames
 - spatial distribution of corresponding horizontal velocity (y-component) to be estimated by the network.

The data files are provided with MEMMAP format defined in the numpy package.

