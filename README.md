# facial-recogntition

The facial recognition algorithm is able to recognize a person from an image. This process combines deep convolutional neural networks and Dlib libraries to identify faces in an image and then classify them with a label [name(s) of that person].

## Getting Started

This code has been tested on Ubuntu 14.04 computer with NVIDIA GTX 980Ti running CUDA 7.5 with CuDNNv4. Follow allong with the instructions and install all dependencies to get your project started.

### Prerequisites

1. Install CUDA (Tested on CUDA 7.5)

	[NVIDIA CUDA Homepage](https://developer.nvidia.com/cuda-toolkit "NVIDIA CUDA Homepage")
	
2. Install cuDNN (Tested on cuDNN 4)

	[NVIDIA cuDNN Homepage](https://developer.nvidia.com/cudnn "NVIDIA cuDNN Homepage")

3. Install OpenCV (Tested on OpenCV 2.4.13)

	* [Download OpenCV 2.4.13](https://github.com/Itseez/opencv/archive/2.4.13.zip "Download OpenCV 2.4.13")

	* [Installation Instructions](http://docs.opencv.org/2.4/doc/tutorials/introduction/linux_install/linux_install.html "Installation Instructions")

4. Install Caffe

	* [Download caffe master branch](https://github.com/BVLC/caffe/archive/master.zip "Download caffe master branch")
	
	* [Installation Instructions](http://caffe.berkeleyvision.org/install_apt.html "Installation Instructions")
	
5. Compile Dlib (Tested on Dlib 19.1)

	* [Download Dlib 19.1 Library](http://dlib.net/files/dlib-19.1.zip "Download Dlib 19.1 Library")

	* Extract the contents and run `python setup.py install` to compile libraries for python.


### Running the scripts

1. Adding people to existing model

	The **create_data_AIA_faces.ipynb** reads from an input root folder, extracts image frames from video files, crops images containing faces, "frontalizes" the face and stores the resultant images to output root folder. It searches for a subfolder containing video of an individual. the subfolder must be named after that individual. You can add multiple videos of people as long as each individual's videos is inside his/her own subfolder. The notebook also creates text files named `train.txt`, `test.txt` and `holdout.txt` that the model uses to train, validate and finally test the performance of the model.

	**Add stages of image processing here. Normal image, cropped image and frontalized image.** 

2. Training model to learn newly added faces

	The **Train_model.ipynb** reads the prepared images from the output root folder and trains the 16-layer VGG Convolutional Neural Network. The weights from the models are saved in the weights folder.

3. Testing model performance on unseen images

	The **Classify.ipnb** uses the latest trained weights to run a forward pass on an unseen image from user defined location and identify the person(s) in the given image.

### Performance

Number of People | Classification Accuracy
------------ | -------------
13 | 99.2%
23 | TBA

### References

[Frontalization code](https://github.com/dougsouza/face-frontalization "Frontalization code")

Tal Hassner, Shai Harel, Eran Paz and Roee Enbar, Effective Face Frontalization in Unconstrained Images, IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), Boston, June 2015
