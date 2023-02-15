# Introduction
Patient monitoring is a crucial aspect of healthcare, as it allows healthcare professionals to closely track a patient's vital signs and detect any potential issues before they become serious. Our Approach makes a vital extraction pipeline that will help to extract vitals and digitize graphs via images from a vital monitoring system. We have tried and used numerous pipelines and several models in our vital extraction system and have tuned our model over the training images.

# Methodology
![Pipeline](https://drive.google.com/uc?id=1j8QBZONjpRvy72rdZm7u7EQs1Xu7Hk4f)
The main pipeline we have applied is text detection and recognition model. For text detection we have used transfer learning over a pretrained YoloV7 model. 

![Yolo Architecture](https://blog.roboflow.com/content/images/2022/07/image-33.webp)

Yolov7 is a single stage object detector from the YOLO family.The author claimed that yolo-v7 effectively reduces about 40% parameters and 50% computation of state-of-the-art real-time object detectors. It focuses on optimizing the training process. In a YOLO model, image frames are featurized through a backbone. These features are combined and mixed in the neck, and then they are passed along to the head of the network YOLO predicts the locations and classes of objects around which bounding boxes should be drawn.
Hyperparameters used in YoloV7:
* Number of epochs: 300
* Warmup Epochs: 3
* Warmup Momentum: 0.8
* Warmup_bias_lr: 0.1
* Momentum: 0.937  # SGD momentum/Adam
* Initial Learning Rate: 0.01

![Easy OCR Framework](https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/easyocr_framework.jpeg)
For the purpose of Text Recognition we have used EasyOCR. The EasyOCR is an open-source end-to-end text detection and recognition system. The main API provided is readtext(), which extracts all the text in a given image. Direct usage of this API was giving unwanted text as well. An exact algorithm to clean the output was infeasible as what unwanted text will be detected was unpredictable. The readtext() API is decoupled into two other APIs, which are detect() and recognize(). Since our YOLO model trained on the given dataset performed pretty well in detecting the vitals and labelling them with their type, we decided to use only the recognize() API. The recognize() API needs a list of detected text bounding boxes. It recognizes only the text that is present in the bounding boxes. This made cleaning the output much easier as now only the vital values, and some surrounding characters such as ‘(‘, ‘)’ and ‘/’ were being recognized. Cleaning this is much easier. Another major reason for using EasyOCR is its architecture. It allows us to integrate our custom models into the pipeline to do recognition. Initially, we had planned to put our custom pipeline mentioned below in it. But as a dataset with vitals and their values was not given and making one in a short span was not feasible hence we didn’t use our pipeline and used the standard pipeline provided which uses accurate models.

For the purpose of Digitizing the graphs,we observed that when we plotted a histogram for pixel distribution over an image there were typically two set of classes foreground and the background. Since we could binarise the image by applying a threshold, we applied OTSU thresholding to maximise the variance between classes and obtain an optimal threshold value where the background and foreground could be distinguished clearly. When the threshold is applied, the wave is visible on a pitch-black background. To digitise graphs, we perform pointwise operations on the entire image to locate all the white pixels and then display the graph while keeping the original aspect ratio.

# Other approaches used:

##### Inferece via Onxx Runtime
Rather than performing the inference in the PyTorch environment, a good alternative can be to do the inference in the ONXX runtime environment. ONXX provides an intermediary representation that optimises various CPUs and GPUs, hence it makes sense to run the Pytorch inference pipeline under the ONXX runtime. The official repository of YOLOv7 contains the script export.py, which can be used to convert a Pytorch model to ONXX format. Next, we create a ONXX session, perform the inference on the converted model, and find that the inference time is a bit higher than the Pytorch inference time. Next, rather than performing inference on a single image, we perform the inference on a batch of images as it might be the case that inference on a batch of images is faster in ONXX than Pytorch. However, we find that the time is almost the same, which is why we continue with the Pytorch version. 

##### Monitor Layout Classification
Given the dataset of 4 monitor layout models, our initial approach was to train a different model for each type of layout monitor. To do this for given a monitor image, we needed to first classify it according to its type. For this, we trained a standard classification model. The dataset was divided into 900 images for training and 100 for testing. The neural network architecture is given below:

![Monitor Image Layout](https://drive.google.com/uc?id=1hXpdZIICgvCm_8JJlSE6YN1WPaGUTcCO)

The model was working pretty decent on the test dataset, giving an F1 score of 0.98 on the test dataset. But in the end, we realised a generalised model for detection trained on all four types of monitors layouts was performing well, and we didn’t need four separate models the classification network was not used.

#####  Number extraction via filter
A rudimentary approach we thought of was applying filters of the colours in which different vital signs were written on the monitor. So, for example, in the above image, each number is written in a different colour, and we can create a filter of that colour and create a binary version of the image where only the digits written in that colour are highlighted. The rest of the image is black. For this approach to work for all the different types of monitors, we tried creating different filters for each type. However, this approach only works in the ideal case when all monitors of a type have the same colour variations. As can be seen from the images in the dataset, they are present in different lighting conditions, due to which no single filter works well for all images of a type of monitor and fails to give a good binary image. We tried to remediate the problem using the Gaussian thresholding approach to reduce the noise in the binarised image, but this approach failed to give the desired results. The accuracy of this step is important as the next step will be to read the digits in the binary image using OCR. Still, if the binary image contains noise and the masking is not proper, then the accuracy of OCR will be adversely affected. Hence, to enforce the accuracy of our entire pipeline, we thought to use an efficient and accurate algorithm to detect the different digits, which is where we went with YOLOv7 as our backbone of the inference pipeline. 

#####  Contour extraction and digit recognition

A different approach we've tried to employ to identify numbers inside a specific bounding box uses contour and easy digit recognition. We first chose this method because we thought it would be faster than EasyOCR, which uses advanced models like LSTM and Resnet. Instead, it makes use of a simple 3-layer model.In order to binarize the image, we first use YOLOV7 to extract a crucial region from a crucial monitor. To find exterior contours in the binarized image, we used contour detection on the original image. We also used additional preprocessing to produce a binary image that was more optimal and provided the desired results in terms of numbers. On the MNIST dataset, we trained a straightforward digit classifier, which we use to determine the precise value of the number we're looking for.This strategy provides us with a fairly accurate forecast, but because we lack a vitals labeled dataset, we are unable to gauge the model's accuracy over a sizable testing set.




