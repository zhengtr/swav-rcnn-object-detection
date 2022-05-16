import os
from PIL import Image
import torch
from torchvision.transforms import functional as F

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

def main():
    #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    unlabeled_data = UnlabeledDataset(root = '/unlabeled', transform = None)
    data_loader = torch.utils.data.DataLoader(unlabeled_data, batch_size=1024,  num_workers=4)

    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in data_loader:
        #data: batch x 3 x 244 x 244// not using the collate fn by prof
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
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