import torch
from dataset import EpisodicSampler, FolderExplorer
from model import get_model
from config import cfg  # cfg.paths is a list of paths to the datasets
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm

def main(cfg):
    folder_explorer = FolderExplorer(cfg.paths)

    paths = folder_explorer()

    sampler = EpisodicSampler(paths = paths,
                              n_query= cfg.sampler.n_queries,
                              n_ways = cfg.sampler.n_ways,
                              n_shot = cfg.sampler.n_shots,)
    
    episode = sampler() #episode is (dataset, classe, support/query, image_path)
    
    imagenet_sample = episode["imagenet"]
    

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False)
    model.to(device)
    model.eval()
    
    transforms = T.Compose([
        T.ToTensor(),
        T.Resize(224, antialias=True), # generates warning otherwise
        T.CenterCrop(224),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225]) # imagenet mean and std
    ])
    
    with torch.inference_mode():
        # compute features for support set
        support_features = {}
        support_images = [image_path for classe in imagenet_sample for image_path in imagenet_sample[classe]["support"]]
        support_classes = [classe for classe in imagenet_sample for image_path in imagenet_sample[classe]["support"]]

        # batch 
        support_batch = [support_images[i:i+cfg.batch_size] for i in range(0,len(support_images),cfg.batch_size)]
        classes_batch = [support_classes[i:i+cfg.batch_size] for i in range(0,len(support_classes),cfg.batch_size)]
        
        for batch, class_batch in tqdm(zip(support_batch,classes_batch), total=len(support_batch)):
            
            images = [Image.open(image_path).convert("RGB") for image_path in batch] # png have 4 channels
            batch_images = torch.stack([transforms(image) for image in images]).to(device) # bs,3,224,224
            
            outputs = model(batch_images)
            
            # unpack outputs
            for i,classe in enumerate(class_batch):
                if classe not in support_features:
                    support_features[classe] = outputs[i].unsqueeze(0)
                else:
                    support_features[classe] = torch.cat([support_features[classe],outputs[i].unsqueeze(0)],dim=0)
        # compute features for query set
        query_features = {}
        query_images = [image_path for classe in imagenet_sample for image_path in imagenet_sample[classe]["query"]]
        query_classes = [classe for classe in imagenet_sample for image_path in imagenet_sample[classe]["query"]]
        
        # batch
        query_batch = [query_images[i:i+cfg.batch_size] for i in range(0,len(query_images),cfg.batch_size)]
        classes_batch = [query_classes[i:i+cfg.batch_size] for i in range(0,len(query_classes),cfg.batch_size)]
        
        for batch, class_batch in tqdm(zip(query_batch,classes_batch), total=len(query_batch)):
            
            images = [Image.open(image_path).convert("RGB") for image_path in batch]
            batch_images = torch.stack([transforms(image) for image in images]).to(device)
            
            outputs = model(batch_images)
            
            # unpack outputs
            
            for i,classe in enumerate(class_batch):
                if classe not in query_features:
                    query_features[classe] = outputs[i].unsqueeze(0)
                else:
                    query_features[classe] = torch.cat([query_features[classe],outputs[i].unsqueeze(0)],dim=0)
                    
        print(support_features.keys())
        print(query_features.keys())
        
        print("J'ai fini ma partie à vous de jouer")
        
        ## TO DO: pass the features to the NCM model
        
        # structure of support_features and query_features:
        # {classe : tensor of features (n_shot or n_query, d)} (tensors on gpu)
        
                
if __name__ == "__main__":
    
    main(cfg)