import numpy as np
import torch
import cv2
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from typing import Dict
import argparse


parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--path', default='', type=str, metavar='P',
                    help='path with images or toward an image (with either JPG, PNG extention)')
parser.add_argument('--save', default=True, type=bool, metavar='S',
                    help='save the output into a json file')
parser.add_argument('--box_nms_thresh', default=0.5, type=float, metavar='B',
                    help='sam parameter ')
parser.add_argument('--min_mask_region_area', default=500, type=int, metavar='M',
                    help='sam parameter ')
parser.add_argument('--area_treshold', type=int, metavar='A',
                    help='min area, if under not consider (in percentage default 1%)')
                    

args = parser.parse_args()


class SAM():
    def __init__(self, path:str, box_nms_thresh=0.5:float, min_mask_region_area=500:int, area_treshold=1:int, save=False:bool):
        
        self.paths = paths
        self.generator = SamAutomaticMaskGenerator(
                            model=sam,
                            box_nms_thresh=box_nms_thresh, # default 0.7 decresea => less duplicates 
                            min_mask_region_area=min_mask_region_area
                        )
        self.predictions = {}
        self.area_treshold = area_treshold
        self.save = save
    
    def get_image(self, path)->np.ndarray:
        """ convert a path into an image """
        image = cv2.imread(t)        
        if image is None:
            print(f"Could not load '{path}' as an image, skipping...")
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    def predict(self, image:np.ndarray, area_treshold:int=self.area_treshold)->Dict:
        """ 
        input : 
        image : the image to segment (np.array) 
        area_treshold : repressent the min percentage of the total area the mask can be by default 1%

        output : 
        processed_image : the mask image - eg the mask with every thing else black
        area : area of the mask 
        bbox : box of the mask to crop 
        """
        assert((str(type(image))!="<class 'torch.Tensor'>")), "Image must be a np.ndarray"

        predition = {}
        total_area = image.shape[0]* image.shape[1]
        # predict the mask of the image 
        masks = self.generator.generate(image)
        for mask in masks: 
            if mask['area'] *100/total_area > area_treshold: 
                image_masked = image.copy()
                image_masked[masks[15]['segmentation'] == False] = [0, 0, 0]  # Set the BGR color to [0, 0, 0] for black
                image_masked = cv2.cvtColor(image_masked, cv2.COLOR_BGR2RGB)
                prediction['processed_image'] = image_masked
                prediction['area'] = mask['area']
                prediction['bbox'] = mask['bbox']
        return predition
        

    def run ():
        if '.jpg' in self.path.lower() or '.png' in self.path.lower():
            image  = self.get_image(self.path)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                continue
            self.predictions[path] =  self.predict(image)

        for image_name in os.listdir(self.path) : 
            if '.jpg' in image_name or '.png' in image_name:  
                path = os.join(path, image_name)
                image  = self.get_image(path)
                if image is None:
                    print(f"Could not load '{t}' as an image, skipping...")
                    continue
                self.predictions[path] =  self.predict(image)
        
        if self.save:
            base = os.path.basename(self.path)
            base = os.path.splitext(base)[0]
            save_base = os.path.join(args.output, base)
            os.makedirs(save_base, exist_ok=False)
            write_masks_to_folder(masks, save_base)


def main():

    sam = SAM(
        args.path, 
        args.box_nms_thresh,
        args.min_mask_region_area, 
        args.area_treshold,
        args.save)
    sam.run()


if __name__ == '__main__':
    main()

           