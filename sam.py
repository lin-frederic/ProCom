import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import matplotlib.pyplot as plt
import cv2
from typing import Dict
import argparse


parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--image_paths', default='', type=str, metavar='P',
                    help='path with images or toward an image (with either JPG, PNG extention)')
parser.add_argument('--model_path', default='', type=str, metavar='F',
                    help='path with images or toward an image (with either JPG, PNG extention)')  
parser.add_argument('--model_type', default='', type=str, metavar='T',
                    help='Type of the model ')  
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
    def __init__(self, image_paths:str, model_path:str, model_type:str, box_nms_thresh:float=0.5, min_mask_region_area:int=500, area_treshold:int=1, save:bool=False):
        assert (torch.cuda.is_available()), "Must have a GPU to run SAM"
        assert (".pth" in model_path), "The model path must lead the the weiths .pth file"
        assert (model_type is not None), "The type of the model must be fed "
        self.image_paths = image_paths

        # load sam 
        print("load sam")
        sam_checkpoint = model_path
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.generator = SamAutomaticMaskGenerator(
                            model=sam,
                            box_nms_thresh=box_nms_thresh, # default 0.7 decresea => less duplicates 
                            min_mask_region_area=min_mask_region_area
                        )
        # init parameter 
        self.predictions = {}
        self.area_treshold = area_treshold
        self.save = save
        print("test")


    def get_image(self, path)->np.ndarray:
        """ convert a path into an image """
        image = cv2.imread(path)        
        if image is None:
            print(f"Could not load '{path}' as an image, skipping...")
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def show_image(self, path):
        plt.figure(figsize=(10,10))
        plt.imshow(self.get_image(path))
        plt.axis('off')
        plt.show()

    def predict(self, image:np.ndarray, area_treshold:int)->Dict:
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

        prediction_list = []
        total_area = image.shape[0]* image.shape[1]
        # predict the mask of the image 
        masks = self.generator.generate(image)
        #print("Total masks" , len(masks))
        for mask in masks: 
            prediction = {}
            if mask['area'] *100/total_area > area_treshold: 
                #image_masked = image.copy()
                #image_masked[mask['segmentation'] == False] = [0, 0, 0]  # Set the BGR color to [0, 0, 0] for black
                #image_masked = cv2.cvtColor(image_masked, cv2.COLOR_BGR2RGB)
                #prediction['processed_image'] = image_masked
                prediction['area'] = mask['area']
                prediction['bbox'] = mask['bbox']
                prediction['segmentation'] = mask['segmentation']

                prediction_list.append(prediction)
        #print("kepts masks" , len (prediction_list))
        return prediction_list
        

    def run (self):
        if '.jpg' in self.image_paths.lower() or '.png' in self.image_paths.lower() or '.jpeg' in self.image_paths.lower():
            image  = self.get_image(self.image_paths)
            if image is None:
                print(f"Could not load '{t}' as an image, skipping...")
                
            self.predictions[self.image_paths] =  self.predict(image, area_treshold=self.area_treshold)
        else : 
            for image_name in os.listdir(self.image_paths) : 
                if '.jpg' in image_name or '.png' in image_name or '.jpeg' in image_name:  
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
    print("run")
    sam = SAM(
        args.image_paths, 
        args.model_path,
        args.model_type,
        args.box_nms_thresh,
        args.min_mask_region_area, 
        args.area_treshold,
        args.save)

    """
    sam = SAM(
            image_paths="/nasbrain/datasets/imagenet/images/val/n01514668/ILSVRC2012_val_00000911.JPEG" , 
            model_path="/nasbrain/f21lin/sam_vit_b_01ec64.pth",
            model_type="vit_b",
            box_nms_thresh=0.5,
            min_mask_region_area=500, 
            area_treshold=1,
            save=False)

    # to show the prediction 
    path = list(sam.predictions.keys())[0]
    mask  = sam.predictions[path][7]["segmentation"]
    plt.figure(figsize=(10,10))
    plt.imshow(mask)
    plt.axis('off')
    plt.show()
    """
    sam.run()


if __name__ == '__main__':
    main()
           
