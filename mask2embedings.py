import embedding
import maskBlocks as mb
from torchvision import transforms as T
import cv2
from PIL import Image
from typing import List
import os
import numpy as np
import json

class mask2embedings():
    def __init__(self, 
                 model:str, 
                 path:str, # todo le changer en list de path 
                 path_to_save:str,  
                 embedding:List[str], 
                 n_imgs:int=None,): 
        self.model_name = model 
        self.path = path
        self.path_to_save = path_to_save

        # list of all images in the file
        if n_imgs is not None : 
            img_paths = np.random.choices(os.listdir(self.paths), k=n_imgs)
        else : # all images 
            img_paths = os.listdir(self.path)

        # list of all the images 
        self.images = []
        self.image_names = []
        for path_image in img_paths:
            self.image_names.append ()# TODO get image name at the end of img_path 
            self.images.append(T.Resize((224,224), antialias=True)(Image.open(path_image).convert("RGB")))
        
        if self.model_name.lower()=="identity":
            self.model = mb.Identity()

        if self.model_name.lower()=="sam":
            self.model = mb.SAM(size="s")

        if self.model_name.lower()=="lost":
            dino_model = mb.get_model(size="s",use_v2=False).to("cuda") # shared model for lost and dsm
            lost_deg_seed = mb.Lost(alpha=1., k=100, model=dino_model)
            lost_atn_seed = mb.Lost(alpha=0., k=100, model=dino_model)
            

        if self.model_name.lower() in ["deepspectral", "dsp", "deep spectral"] :
            dino_model = mb.get_model(size="s",use_v2=False).to("cuda") # shared model for lost and dsm
            self.model = mb.DeepSpectralMethods(model=dino_model, n_eigenvectors=5)

        else : 
            print("Need a valid model name either :identity, lost , sam of dsp ")
            self.model = None
    

    def generate_masks (self):
        assert (self.model is not None), "The model shouldn't be none"

    def save_masks(self):
        """ the mask are save in a json file with their embedded image"""
        img2mask = {}
        for index , img in self.images_names :
            img2mask[img] = {'mask' : self.model(self.images[index])} 

        with open( ): # TODO 
            json.dump(img2mask)
        print(f"Mask and images generated with {self.model} saved at {self.path_to_save}")

    def save_maskEmbedded(self):
        """ embed the mask ans save them """
        print(f"the embedded mask images generated with {self.model} saved at {self.path_to_save}")