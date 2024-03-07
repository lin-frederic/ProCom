# :herb: FICUS: Few-shot Image Classification with Unsupervised object Segmentation


Official code repository for EUSIPCO 2024 paper 
"[FICUS: Few-shot Image Classification with Unsupervised object Segmentation](https://.pdf)". 

The paper is available at [https:/lien eusipco.pdf](https://.pdf).

[IMT Atlantique](https://www.imt-atlantique.fr/en) 
Jonathan Lys, Frédéric Lin, Clément Beliveau, Jules Decaestecker 
[Lab-STICC](https://www.imt-atlantique.fr/fr/recherche-innovation/communaute-scientifique/organismes-nationaux/lab-sticc)
Yassir Bendou, Aymane Abdali,\Bastien Pasdeloup

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)
[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
<CENTER>
<img
src="https://www.pole-emc2.fr/app/uploads/logos_adherents/91fff3f6-c993-67c6-68ae-53957c2f623d-768x522.png"
WIDTH=200 HEIGHT=200>
</CENTER>

This repository contains the code for out of the box ready to use few-shot classifier for ambiguous images. In this paper we have shown that removing the ambiguity from the the query during few shot classification improves performances. To do so we use a combination of foundation models and spectral methods. 
## Installation 🛠 

### Conda venv

```[bash]
   git clone https://github.com/expertailab/.git
   cd 
   python3 -m venv ~//venvFicus
   source ~/venvFicus/bin/activate
   pip install -r requirement.txt
```
### Conda env 

```[bash]
   git clone https://github.com/expertailab/.git
   cd 
   conda create -n Ficus python=3.9
   conda activate Ficus
   pip install -r requirement.txt
```
## Pipeline 



## Get started 🚀

### Dataset 

For all our experiments we have used three datasets  : ImageNet , Pascal Voc and Cub
### Models 

We use two foundation model : `dinov2_vit{s-b-l-g}14` for image embdedding and classification and [Segment Anything ](https://github.com/facebookresearch/segment-anything) for image segmentation.

### Run inference

- To run the evaluations  
```[bash]
sh run.sh -dataset "cub"
```
- To run deep spectral method on un image
```[bash]

```

Expected result : 
<CENTER>
<img
src="https://https://github.com/NewS0ul/ProCom/blob/main/images/figs/Asample_points_2010_000805.jpg.png"
WIDTH=200 HEIGHT=200>
</CENTER>

- To run prompted sam on an image
```[bash]

```

Expected result : 
<CENTER>
<img
src="https://https://github.com/NewS0ul/ProCom/blob/main/images/figs/Asample_points_2010_000805.jpg.png"
WIDTH=200 HEIGHT=200>
</CENTER>

- To run ficus on an image 
```[bash]

```


## Citation

If you find our paper or code repository helpful, please consider citing as follows:

```
bibtex
```
