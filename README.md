# Distributed CNN training #
This is a reference implementation of distributed CNN training using fusing, tiling and grouping. It is built upon the original darknet framework. The technique targets early feature dominated convolutional and pooling layers.

# Implementation and test run #
The implementation is added as a feature onto the darknet framework. The core implementation of the technique is under `src/ftp.c`. To enable the feature, under the Makefile, the `FTP` option can be set to 1 and built. There is a configuration file `cfg/ftp.cfg` where the parameters for the distributed training setup can be configured, namely the number of tiling grid, number of initial layers, device IP addresses, and the grouping profiles.

# Future work #
Currently, the implementation targets the early layers and is just a prototype implementation. Future work will explore distributed techniques for the later layers which are generally weight dominated to train on the complete network. 

# Contact
If you have any questions or want to discuss anything about this, author can be contacted at 
* Pranav Rama - pranavrama9999@utexas.edu