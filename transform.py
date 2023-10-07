from torchvision import transforms as T
from torchvision.transforms import functional as F

class PadAndResize:
    """
    Pad and resize an image to a target size
    Similar to SAM's ResizeLongestSide
    
    Parameters:
        target_size (int) : target size of the image (image will be squared)
        pad_value (int): value to pad the image with (between 0 and 255), default is 0
    Args:
        image (PIL image): image to pad and resize (RGB)
    Returns:
        tensor: tensor of shape (3, target_size, target_size)
    
    """
    def __init__(self, target_size, pad_value=0) -> None:
        assert isinstance(target_size, int) and target_size > 0, "target_size should be a positive int"
        assert isinstance(pad_value, int) and 0 <= pad_value <= 255, "pad_value should be an int between 0 and 255"
            
        self.target_size = target_size
        self.pad_value = pad_value
        self.to_tensor = T.ToTensor() # convert PIL image to tensor at the end (also normalize)
        
    def __call__(self, image):
        # image is a PIL image
        w, h = image.size
        
        new_w, new_h = self.target_size, self.target_size
            
        # resize
        max_size = max(w,h)
        ratio = max_size / self.target_size
        new_w = int(w / ratio)
        new_h = int(h / ratio)
        
        image = F.resize(image, (new_h,new_w))
        
        # pad
        
        delta_w = self.target_size - new_w
        delta_h = self.target_size - new_h
        
        # compute padding
        left = delta_w // 2
        right = delta_w - left
        top = delta_h // 2
        bottom = delta_h - top
        
        image = F.pad(image, (left, top, right, bottom), self.pad_value)
        
        # convert to tensor
        image = self.to_tensor(image)
        
        return image