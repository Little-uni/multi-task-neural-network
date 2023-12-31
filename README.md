# multi-task-neural-network
Core codes for the multi-task neural network paper

**1. System requirements**

**1.1 All software dependencies and operating systems (including version numbers):** 

Windows 10 (Version 21H1).

**1.2 Versions the software has been tested on:** 

Python 3.7.

**1.3 Any required non-standard hardware:**

No.

**2. Installation guide**

**2.1 Instructions:** 

The example code can be run on Platforms of Python 3.6 (or higher).

**2.2 Typical install time on a "normal" desktop computer:** 

1 hour.

**3. Demo**
**3.1 Datasets:** 

This repository contains two datasets: "REE dataset_10% sample.xlsx" and "REE dataset_Franus.xlsx."

The first dataset provides 10% of the raw samples used to train the multi-task neural network models. We provide these samples to assist in 1) running the demo analysis, 2) understanding the architecture of our multi-task neural network model, and 3) realizing the pipeline of the model prediction. However, it's essential to note that this demo may not achieve the same level of prediction accuracy, as model training is dependent on the dataset's population. We can consider disclosing the other samples based on reasonable inquiry.

The second dataset is a validation dataset used in the transfer learning analysis. This dataset is curated based on the data reported by Franus et al. (doi: 10.1007/s11356-015-4111-9).

**3.2 Instruction for Running the Codes:**

This repository contains two codes: "Model training.py" and "Predict with the pretrained models.py."

The first code is a pipeline for training both single-task and multi-task neural network models (which are comparatively studied in this work). To run this code, download the dataset, "REE dataset_10% sample.xlsx," and update its path in Line 152.

The second code is used to load the trained models in this study for making predictions. These include both the pretrained multi-task neural network models and the ones transferred based on Franus' dataset. To run this code, download the dataset, "REE dataset_Franus.xlsx," and update its path in Line 68. Additionally, download the models and scalers, updating their paths in Lines 99, 105, 111, and Lines 73 and 74, respectively.

**3.3 Expected run time:** 

5-10 minutes — we suggest the reader run this demo on Google Colab to avoid potential issues with software installation and environment setup.

