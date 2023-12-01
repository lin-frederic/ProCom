import torch
from dataset import EpisodicSampler, FolderExplorer
from model import get_model
from config import cfg  # cfg.paths is a list of paths to the datasets
from ncm import NCM
from PIL import Image
from torchvision import transforms as T
from tqdm import tqdm
from transform import PadAndResize
import numpy as np


def main(cfg):
    folder_explorer = FolderExplorer(cfg.paths)

    paths = folder_explorer()

    sampler = EpisodicSampler(paths = paths,
                              n_query= cfg.sampler.n_queries,
                              n_ways = cfg.sampler.n_ways,
                              n_shot = cfg.sampler.n_shots,)
    L_acc = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(size="s",use_v2=False)
    model.to(device)
    model.eval()
    
    pbar = tqdm(range(cfg.n_runs), desc="Runs")
    for episode_idx in pbar:
        
        # new sample for each run
        episode = sampler() #episode is (dataset, classe, support/query, image_path)

        imagenet_sample = episode["imagenet"]

        transforms = T.Compose([
            #PadAndResize(224), # pad and resize to 224x224, to_tensor
            T.Resize((224,224)),
            T.ToTensor(),
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
            
            for batch, class_batch in zip(support_batch,classes_batch):
                
                images = [Image.open(image_path).convert("RGB") for image_path in batch] # png have 4 channels
                batch_images = torch.stack([transforms(image) for image in images]).to(device) # bs,3,224,224
                
                outputs = model(batch_images)
                
                # unpack outputs
                for i,classe in enumerate(class_batch):
                    if classe not in support_features:
                        support_features[classe] = {"features": torch.empty(0,dtype=torch.float32,device=device),
                                                    "indices": torch.empty(0,dtype=torch.int64,device=device)}
                    support_features[classe]["features"] = torch.cat((support_features[classe]["features"],outputs[i].unsqueeze(0)),dim=0)
                    index = len(support_features[classe]["features"])-1
                    support_features[classe]["indices"] = torch.cat((support_features[classe]["indices"],
                                                                    torch.tensor([index],dtype=torch.int64,device=device)),dim=0)
                    
                    
            # compute features for query set
            query_features = {}
            query_images = [image_path for classe in imagenet_sample for image_path in imagenet_sample[classe]["query"]]
            query_classes = [classe for classe in imagenet_sample for image_path in imagenet_sample[classe]["query"]]
            
            # batch
            query_batch = [query_images[i:i+cfg.batch_size] for i in range(0,len(query_images),cfg.batch_size)]
            classes_batch = [query_classes[i:i+cfg.batch_size] for i in range(0,len(query_classes),cfg.batch_size)]
            
            for batch, class_batch in zip(query_batch,classes_batch):
                
                images = [Image.open(image_path).convert("RGB") for image_path in batch]
                batch_images = torch.stack([transforms(image) for image in images]).to(device)
                
                outputs = model(batch_images)
                
                # unpack outputs
                
                for i,classe in enumerate(class_batch):
                    if classe not in query_features:
                        query_features[classe] = {"features": torch.empty(0,dtype=torch.float32,device=device),
                                                    "indices": torch.empty(0,dtype=torch.int64,device=device)}
                    query_features[classe]["features"] = torch.cat((query_features[classe]["features"],outputs[i].unsqueeze(0)),dim=0)
                    index = len(query_features[classe]["features"])-1
                    query_features[classe]["indices"] = torch.cat((query_features[classe]["indices"],
                                                                    torch.tensor([index],dtype=torch.int64,device=device)),dim=0)

            
            ncm = NCM()
            run_acc = ncm(support_features, query_features)
            L_acc.append(run_acc)
            pbar.set_description(f"Run acc: {100*run_acc:.3f}, avg acc: {100*np.mean(L_acc):.3f}")
            
    avg_acc = np.mean(L_acc)
    std_acc = np.std(L_acc)
    print(f"\nacc over {cfg.n_runs} runs: {100*avg_acc:.3f} +- {100*std_acc:.3f}")
    chance = 1 / cfg.sampler.n_ways["imagenet"]
    print("kappa: ", (avg_acc - chance) / (1 - chance))
    print(np.round(L_acc,3))
    # structure of support_features and query_features:
    # {classe : {"features": tensor([n_shot, d]), "indices": tensor([n_shot])}, ...
    #  classe : {"features": tensor([n_query, d]), "indices": tensor([n_query])}, ...}
    # put indices because we need to know which feature corresponds to which image when including the masks
    
                
if __name__ == "__main__":
    
    main(cfg)