## SDM: A Matlab implementation of Supervised Descent Method for facial landmark detection and tracking

### Resources

* My Homepage: <https://sites.google.com/site/zhenhuaswebpage/home>

* Please cite the following publications if you use this code.

1. Feng, Z. H., Huber P., Kittler J., Christmas W. & Wu X. J. Random cascaded-regression copse for robust facial landmark detection. IEEE Signal Processing Letters, 2015, 1(22), pp:76-80.

2. Feng, Z. H., Hu G., Kittler J., Christmas W. & Wu X. J. Cascaded collaborative regression for robust facial landmark detection trained using a mixture of synthetic and real images with dynamic weighting. IEEE Trans. on Image Processing, 2015, 24(11), pp:3425-3440.

3. Xiong, X., & De la Torre, F. Supervised descent method and its applications to face alignment. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2013, pp:532-539.

### Guide for use

1. Create a folder with name 'data' for storing training and test data, and a folder with name 'model' for storing a trained model, under the main directory

2. Download the COFW color images from http://www.vision.caltech.edu/xpburgos/ICCV13/ and unzip the .mat files to the 'data' folder

3. Run the example_detection.m code for SDM training and test for facial landmark detection

* Notice: The code was tested on Matlab 2016a with the Computer Vision System Toolbox. If you do not have this tool box. You can use the vlfeat toolbox instead.

### Contact

Dr. Zhenhua Feng

Centre for Vision, Speech and Signal Processing, University of Surrey

z.feng@surrey.ac.uk, fengzhenhua2010@gmail.com
