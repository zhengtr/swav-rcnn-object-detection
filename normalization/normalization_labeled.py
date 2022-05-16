import os
from PIL import Image
import torch
from torchvision.transforms import functional as F 
from dataset import LabeledDataset
import transforms as T
import utils

class UnlabeledDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """
        self.transform = transform
        self.image_dir = root
        self.filelist = os.listdir(self.image_dir)
        problem_img = ['15460.PNG','151438.PNG', '158432.PNG']
        for p in problem_img:
            if p in self.filelist:
                self.filelist.remove(p)
        self.num_images = len(self.filelist)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # the idx of unlabeled image is not consecutive
        with open(os.path.join(self.image_dir, self.filelist[idx]), 'rb') as f:
            try:
                img = Image.open(f).convert('RGB')
            except OSError:
                print(self.filelist[idx],'caused error')
        return F.to_tensor(img)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())

    return T.Compose(transforms)

def main():
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    train_dataset = LabeledDataset(root='../../data/labeled', split="training", transforms=get_transform(train=True))
    val_dataset = LabeledDataset(root='../../data/labeled', split="validation", transforms=get_transform(train=True))
    dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    print(f'Dataset size: {len(dataset)}')
    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in data_loader:
        #data: batch x 3 x 244 x 244// not using the collate fn by prof
        data = data[0]
        batch_samples = 1
        data = data.view(batch_samples, data.size(0), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        if nb_samples % 1000 == 0:
            print(nb_samples,'samples succeeded!')

    mean /= nb_samples
    std /= nb_samples
    print('mean: ', mean, 'std: ', std)


if __name__ == "__main__":
    main()