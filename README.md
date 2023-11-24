# multi-task-neural-network
Core codes for the multi-task neural network paper

1. System requirements

1.1 All software dependencies and operating systems (including version numbers): 
Windows 10 (Version 21H1).

1.2 Versions the software has been tested on: 
Python 3.7.

1.3 Any required non-standard hardware: 
No.

2. Installation guide

2.1 Instructions: 
The example code can be run on Platforms of Python 3.6 (or higher).

2.2 Typical install time on a "normal" desktop computer: 
1 hour.

3. Demo
3.1 Instructions to run on data:
This fold contains a Excel file containing 10% of the raw samples ("REE dataset_10% sample.xlsx") that were used for modeling. Our intention of providing those samples to help the reader to 1) run the demo analysis, 2) understand the architecture of our multi-task neural network model, and 3) realize the pipeline of the model prediction. It should be noted that, however, this demo does not yield the same level of prediction accuracy, as the machine learning model training is related to the population of the dataset. We can consider disclosing the other samples based on reasonable inquiry.
 
3.2 Expected output: 
A sample comparison of the prediction accuracies between a single-task model and a multi-task model (which is comparatively studied in this paper).

3.3 Expected run time for demo on a "normal" desktop computer: 
5-10 mins.

4. Instructions for use
How to run the software on your data: 
Please download the data file ("REE dataset_10% sample.xlsx") and update its address at Line 162 in the code. We suggest the reader running this demo on Google Colab to avoid potential issues with the software installation and environment setup.
