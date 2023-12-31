# multi-task-neural-network
Core codes for the multi-task neural network paper

&nbsp;&nbsp; **1. System requirements**

&nbsp;&nbsp;&nbsp;&nbsp; **1.1 All software dependencies and operating systems (including version numbers):** 
&nbsp;&nbsp;&nbsp;&nbsp; Windows 10 (Version 21H1).

&nbsp;&nbsp;&nbsp;&nbsp; **1.2 Versions the software has been tested on:** 
&nbsp;&nbsp;&nbsp;&nbsp; Python 3.7.

&nbsp;&nbsp;&nbsp;&nbsp; **1.3 Any required non-standard hardware:**
&nbsp;&nbsp;&nbsp;&nbsp; No.

&nbsp;&nbsp; **2. Installation guide**

&nbsp;&nbsp;&nbsp;&nbsp; **2.1 Instructions:** 
&nbsp;&nbsp;&nbsp;&nbsp; The example code can be run on Platforms of Python 3.6 (or higher).

&nbsp;&nbsp;&nbsp;&nbsp; **2.2 Typical install time on a "normal" desktop computer:** 
&nbsp;&nbsp;&nbsp;&nbsp;1 hour.

&nbsp;&nbsp; **3. Demo**

&nbsp;&nbsp;&nbsp;&nbsp; **3.1 Datasets:** 
&nbsp;&nbsp;&nbsp;&nbsp; This repository contains two datasets: "REE dataset_10% sample.xlsx" and "REE dataset_Franus.xlsx."

&nbsp;&nbsp;&nbsp;&nbsp; The first dataset provides 10% of the raw samples used to train the multi-task neural network models. We provide these samples to assist in 1) running the demo analysis, 2) understanding the architecture of our multi-task neural network model, and 3) realizing the pipeline of the model prediction. However, it's essential to note that this demo may not achieve the same level of prediction accuracy, as model training is dependent on the dataset's population. We can consider disclosing the other samples based on reasonable inquiry.

&nbsp;&nbsp;&nbsp;&nbsp; The second dataset is a validation dataset used in the transfer learning analysis. This dataset is curated based on the data reported by Franus et al. (doi: 10.1007/s11356-015-4111-9).

&nbsp;&nbsp;&nbsp;&nbsp; **3.2 Instruction for Running the Codes:**

&nbsp;&nbsp;&nbsp;&nbsp; This repository contains two codes: "Model training.py" and "Predict with the pretrained models.py."

&nbsp;&nbsp;&nbsp;&nbsp; The first code is a pipeline for training both single-task and multi-task neural network models (which are comparatively studied in this work). To run this code, download the dataset, "REE dataset_10% sample.xlsx," and update its path in Line 152.

&nbsp;&nbsp;&nbsp;&nbsp; The second code is used to load the trained models in this study for making predictions. These include both the pretrained multi-task neural network models and the ones transferred based on Franus' dataset. To run this code, download the dataset, "REE dataset_Franus.xlsx," and update its path in Line 68. Additionally, download the models and scalers, updating their paths in Lines 99, 105, 111, and Lines 73 and 74, respectively.

&nbsp;&nbsp;&nbsp;&nbsp; **3.3 Expected run time:** 
5-10 minutes — we suggest the reader run this demo on Google Colab to avoid potential issues with software installation and environment setup.

