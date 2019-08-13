# Semantic Segmentation (Advanced Deep Leaning Elective)
### Introduction
In this project, the pixels of a road in images are labeled using a Fully Convolutional Network (FCN). More concretely, the pixel of the provided image are labeled as either
* being part of the road
* NOT being part of the road
This way, the network is inteded to help find out which parts of the image are safe for an autonomous vehicle to drive on.


### Approach
##### Provided (Input)
The encoding stage of a pre-trained VGG-16 network is provided.
In the project, a 1x1 convolution complemented by the decoder stage, that is, a 1x1 convolution stage followed by transpose convolutions are implemented. In addition, skip layers are added to improve the performance of the algorithm.

### Architecture
The project uses a Tensorflow AdamOptimizer with a Cross Entropy loss function.

### Training
In order to determine optimal results, I implemented a simple procedure to determine the optimal parameters, that is learning rate and keep probability. Initially, some reasonable values working satisfactorily have been found (50 epochs, batch size 10, keep_prob 0.8 and learning rate 0.001, respectively) simply by trial and error.

Subsequently each parameter has been altered, the outcome checked, and henceforth either increased or decreased depending on the outcome.

The parameters finally found to provide the lowest loss ultimately are:
<font color="red">

* no epochs: 50
* batch_size: 10
* keep_probability: 0.5
* learning_rate: 0.0005
</font>

### Infrastructure
Implementation and testing has been carried out on a custom machine rather than AMI.
The relevant hardware specifications are as follows:
* CPU: AMD Ryzen 7 (8/16 Cores/Threads)
* RAM: 32 GB
* GPU: nVidia GTX 1070
* Graphics Memory: 8GB
* Compute Capability: 6.1

### Results
The average loss after the respective number of epochs:
<font color="red">

* 5 : 0.3
* 10: 0.12
* 20: 0.05
* 30: 0.07
* 40: 0.037
* 50: 0.031
</font>

### Sample Images
Find below some randomly chosen sample images. As can be seen, the network is satisfactorily classifying the pixels as road/not road. That said, the algorithm is certainly not perfect, occasionally classifying erroneously.

![](runs/1504308542.773357/um_000018.png)
![](runs/1504308542.773357/um_000068.png)
![](runs/1504308542.773357/umm_000012.png)
![](runs/1504308542.773357/umm_000062.png)
![](runs/1504308542.773357/uu_000021.png)