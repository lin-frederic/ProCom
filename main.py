import torch
from dataset import EpisodicSampler, FolderExplorer
from model import get_model
from config import cfg  # cfg.paths is a list of paths to the datasets
from PIL import Image
from torchvision import transforms
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folder_explorer = FolderExplorer(cfg.paths)

    paths = folder_explorer()

    sampler = EpisodicSampler(paths = paths,
                              n_query= cfg.sampler.n_queries,
                              n_ways = cfg.sampler.n_ways,
                              n_shot = cfg.sampler.n_shots,)
    model = get_model(size="s",use_v2=False)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for i in range(1):
            episode = sampler() #episode is (dataset, classe, support/query, image_path)
            for dataset in episode:
                for classe in episode[dataset]:
                    for image_path in episode[dataset][classe]["support"]:
                        image = Image.open(image_path).convert("RGB")
                        image = transforms.ToTensor()(image)
                        image = transforms.Resize(224)(image)
                        image = transforms.CenterCrop(224)(image)
                        image = image.unsqueeze(0) # [1, 3, 224, 224]
                        image = image.to(device)
                        print(image.shape)
                        output = model(image)
                        print(output.shape)
                        exit()
                        #TO DO : test ncm with support /query features
if __name__ == "__main__":
    main()