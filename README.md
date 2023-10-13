# PROCOM: Instance-wise features is all you need 🔥


## Introduction 🚀

The goal of this project is to be able to disambiguate the potentially misleading images to improve classification performances. Indeed, whenever the image starts to contain several objects, the classification task becomes trickier as the model has to choose between the different objects detected. Let's imagine an image of a cat sitting next to a dog, this image can be classified either with "dog" label or "cat" label, but the model doesn't know which part referred to a cat and which referred to a dog. 

To remove this uncertainty, the goal is to remove the other objects in a training image and only focus on the part which contains the desired object of the class. To do so, we take a batch of images belonging to the same class and use a class-agnostic object detector to identify all the possible objects within an image. Then, between all the objects detected in the image, we have to find the object which best represents the class. To do so, we extract features from the image, like an attention mask for instance. Then a similarity method is used between those features with all the other features extracted from the other images of the batch. 

The underlying hypothesis is that the batch is big enough to contain enough images without any ambiguities to correctly describe the class. The second hypothesis is that the similarity between the features of those easy images and the corresponding object of the multi-object image is maximal, and is discriminative enough compared to the similarity obtained with unrelated objects. 

At the end of the process, we obtain a denoised dataset which precisely describes a class. 

### Problematic: 



### Approach : 

Two approaches have been proposed, the difference between those approaches relies is the agnostic object detector. One uses LOST which directly detects objects of "interest" in an image. The other approach uses SAM which computes a mask of all the elements present in images: is "segment everything". 

## Schemes of the two options 

The first approach is using [LOST](https://arxiv.org/pdf/2109.14279.pdf)
<img src="images/LOST_pipeline.png"/>
The second approach is using [SAM](https://arxiv.org/pdf/2304.02643.pdf)
<img src="images/SAM_pipeline.png"/>


## Instalation 🛠 : 

To connect to the SSH bridge : 
```

PLEASE FILL 

```

If you don't have conda please install it [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
To install the correct dependencies and set up the environment with conda 

```
conda env create -f environment.yml
conda actiavate ProCom

```


## Gantt Chart 🗓️


```mermaid
gantt

dateFormat  YYYY-MM-DD
title PROCOM

section Development
Environment setup         :active,  des1, 2023-10-05, 21d
Dataset collect/process   :active,  des2, 2023-10-05, 21d
Model instalation         :done,    des3, 2023-10-05, 7d
Baseline implementation   :  des14, 2023-10-19, 7d
Pipeline design     :  des15, 2023-10-19, 7d
Blabla implementation :  des16, 2023-10-26, 7d

section Research
State of Art collect      :active,  des4, 2023-09-29, 50d
Benchmark collect         :active,  des5, 2023-09-29, 50d


section Writing 
Abstract : des17, 2023-10-26, 7d
Introduction : des18, 2023-10-12, 30d
Related Work : des19, 2023-10-12, 30d
Experiement - Dataset : des20, 2023-10-26, 7d
Experiement - Model : des21, 2023-11-02, 7d


section School Project Review
Team organisation        : done, des6, 2023-09-29, 10d        
Subject reformulation    : active, des11, 2023-09-29, 21d
Project planning         : active, des12, 2023-09-29, 7d
Risk evaluation          :        des13, 2023-10-12, 7d


```


## KIKIMETER 📈

<!-- BEGIN MERMAID -->
    
```mermaid
pie
title Number of line of codes per user
"Fred" : 100
"Jules" : 100
"Jonathan" : 100
"Clément" : 0
```

<!-- END MERMAID -->
