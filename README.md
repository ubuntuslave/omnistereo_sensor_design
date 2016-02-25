<img src="https://raw.githubusercontent.com/ubuntuslave/omnistereo_sensor_design/master/images/graphical_representation_of_the_paper.png" title='Omnistereo Sensor Paper Graphical Representation' width='300px' alt='Omnistereo Sensor Paper Graphical Representation' />

# Supplementary Materials for the "*Design and Analysis of a Single-Camera Omnistereo Sensor for Quadrotor Micro Aerial Vehicles (MAVs)*" 

The following iPython Notebooks reproce the analysis plots performed over the *Theoretical Model* for the catadioptric omnidirectional stereo sensor proposed in the corresponding [Sensors Article](http://www.mdpi.com/1424-8220/16/2/217/) published in February 6, 2016.

#### [Demonstration and Analysis Plots](https://nbviewer.jupyter.org/github/ubuntuslave/omnistereo_sensor_design/blob/master/HyperCata%20Demo.ipynb)

#### [Derivation of Analytical Solutions to Projection and Back-Projection of Theorical Omnistereo Model](https://nbviewer.jupyter.org/github/ubuntuslave/omnistereo_sensor-design/blob/master/Hyperbolic%20Rig.ipynb)

- **Author**: Carlos Jaramillo
- **Contact**: <cjaramillo@gradcenter.cuny.edu>

*Copyright (C)* 2016 under the *Gnu Public License version 2 (GPL2)*

***WARNING***: This code is at an experimental stage
 
## Project Overview

`omnistereo_sensor_design` is a *proof-of-concept* written in Python for depth perception using a custom-designed catadioptric omnidirectional stereo sensor.

## Program Setup in Python 3:

Using Python 3.5:

    $ pip3 install numpy
    $ pip3 install scipy
    $ pip3 install sympy

For visualization:

    $ pip3 install matplotlib
    $ pip3 install mpldatacursor

### (Optional) For 3D Visualization:

    $ pip3 install vispy

or 

    $ pip3 install visvis
    

### (Required) OpenCV 3

The following guide uses `Homebrew `for Mac OS X:

#### Requirements:

Make sure you have installed the XCode CLI tools are all installed:

    $ ls -lah /usr/include/sys/types.h 

If not, try:

    $ xcode-select --install

The last release of OpenCV's [Deep Neural Network module](http://docs.opencv.org/master/d6/d0f/group__dnn.html) requires Google's [Protocol Buffers](https://developers.google.com/protocol-buffers/)    

    $ brew install protobuf    
    
However, it *may fail* to find a "suitable threading library available." If so, disable DNN within the CMake configuration (in the next steps)

#### Optional requirement:
    
Install `Qt4` from Homebrew. Then, to build `OpenCV` with `Qt4` and `Python 3.5`, try without the GUI configuration:  
    
    $ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -DWITH_QT=4 -D BUILD_opencv_python3=ON -D PYTHON3_EXECUTABLE=/usr/local/bin/python3 -D PYTHON3_PACKAGES_PATH=lib/python3.5/site-packages -D PYTHON3_LIBRARY=/usr/local/Cellar/python3/3.5.1/Frameworks/Python.framework/Versions/3.5/lib/libpython3.5m.dylib -D PYTHON3_INCLUDE_DIR=/usr/local/Cellar/python3/3.5.1/Frameworks/Python.framework/Versions/3.5/include/python3.5m -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_EXAMPLES=OFF -D OPENCV_EXTRA_MODULES_PATH=/Users/carlos/src/opencv_contrib/modules -D BUILD_opencv_viz=OFF ..

Then, install [PyQT](http://www.riverbankcomputing.co.uk/software/pyqt/intro)

    $ brew install pyqt --with-python3

It can be tested from  a Python Shell:
    
    >>> from PyQt4.QtGui import *

#### Install OpenCV 3 via Homebrew
    
    $ brew install opencv3 --with-contrib --with-ffmpeg --with-gphoto2 --with-gstreamer --with-jasper --with-libdc1394 --with-openni2 --with-python3 --with-qt --with-tbb
    
    $ brew link --overwrite --force opencv3

#### The Hard (more powerful) way from Source Code:
    
Clone *OpenCV 3* from the repo

    $ cd ~/src
    $ git clone git@github.com:Itseez/opencv.git

And also the ***contributed*** modules:

    $ git clone git@github.com:Itseez/opencv_contrib.git    
    
Then, configure the installation: 

    $ cd opencv
    $ mkdir build
    $ cd build
    $ ccmake ../

Configure OpenCV via CMake:

   
Configure options, among the very important

     OPENCV_EXTRA_MODULES_PATH        /blabla...bla/src/opencv_contrib/modules    
     
The following module was causing trouble:

    - WITH_OPENGL
    
Trouble from `opencv_contrib`, so I turned it off:

    - BUILD_opencv_cvv

The *Python 3.5* configuration looked like this:

    PYTHON2_EXECUTABLE               /usr/local/bin/python2.7
    PYTHON2_INCLUDE_DIR              /usr/local/Frameworks/Python.framework/Versions/2.7/include/python2.7
    PYTHON2_INCLUDE_DIR2
    PYTHON2_LIBRARY                  /usr/local/Frameworks/Python.framework/Versions/2.7/lib/libpython2.7.dylib
    PYTHON2_LIBRARY_DEBUG
    PYTHON2_NUMPY_INCLUDE_DIRS       /usr/local/lib/python2.7/site-packages/numpy/core/include
    PYTHON2_PACKAGES_PATH            lib/python2.7/site-packages
    PYTHON3_EXECUTABLE               /usr/local/bin/python3
    PYTHON3_INCLUDE_DIR              /usr/local/Frameworks/Python.framework/Versions/3.5/include/python3.5m
    PYTHON3_INCLUDE_DIR2
    PYTHON3_LIBRARY                  /usr/local/Frameworks/Python.framework/Versions/3.5/lib/libpython3.5.dylib
    PYTHON3_LIBRARY_DEBUG
    PYTHON3_NUMPY_INCLUDE_DIRS       /usr/local/lib/python3.5/site-packages/numpy/core/include
    PYTHON3_PACKAGES_PATH            lib/python3.5/site-packages

The OpenNi configuration (If Enabled, set them by toggling the advanced mode):

    OPENNI2_INCLUDES                /usr/local/include/ni2
    OPENNI2_LIBRARY                 /usr/local/lib/ni2/lib/OpenNI2.dylib
    OPENNI_INCLUDES                 /usr/local/include/ni
    OPENNI_LIBRARY                  /usr/lib/libOpenNI.dylib

which will set something like this:

    OPENNI2_INCLUDE_DIR             /usr/local/include
    OPENNI2_LIB_DIR                 /usr/local/lib/ni2/lib
    OPENNI_INCLUDE_DIR              /usr/local/include
    OPENNI_LIB_DIR                  /usr/lib
    OPENNI_PRIME_SENSOR_MODULE      /usr/lib/libXnCore.dylib
    OPENNI_PRIME_SENSOR_MODULE_BIN  /usr/lib
 
You can ignore any warning about QT5

Compile, and install as usual:

    $ make
    $ make install
 
    
### (Optional) PCL:

See instructions to install PCL and 

#### `python-pcl`

First, get `cython`

    $ pip3 install cython
    
Using [python-pcl](https://github.com/strawlab/python-pcl) 

    $ cd ~/src
    $ git clone https://github.com/strawlab/python-pcl.git
    $ cd python-pcl
    $ python setup.py install

In Python 3, 

    $ python3 setup.py install
    
When attempting a reinstall do this:

    $ pip uninstall python-pcl
    $ make clean
    $ make all
    $ python setup.py install
        

Finally, just make sure the Python's PCL package exists as `/usr/local/lib/python2.7/site-packages/pcl`   
    
    $ pip show python-pcl

or

    $ pip3 show python-pcl    

If problems, 

- Open a new terminal sesion and check with `pip` again.

- Check what this command says:

        $ otool -L /usr/local/lib/python2.7/dist-packages/pcl/_pcl.so | grep libpcl

##### Caveat:
Python-PCL doesn't support color, only types `PointXYZ`. Using C++, the PointCloud and the Colors can be registered (merged) to be become of type `PointXYZRGB`
