import argparse
from dataset import UnlabeledDataset, LabeledDataset
from engine import train_one_epoch, evaluate
import torch
import utils
import transforms as T

parser = argparse.ArgumentParser(description="Evaluate models: Fine-tuning with labeled dataset")
parser.add_argument("--data_path", type=str, default="/labeled",
                    help="path to imagenet")
parser.add_argument("--model_file", type=str, default="model_final.pth",
                    help="path to imagenet")
                    
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():

    global args
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    index = list(range(10))
    valid_dataset = LabeledDataset(root=args.data_path, split="validation", transforms=get_transform(train=False))
    # valid_dataset = torch.utils.data.Subset(valid_dataset, index)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)

    model = torch.load(args.model_file)
    model.eval()
    model.to(device)

    evaluate(model, valid_loader, device=device)

if __name__ == "__main__":
    main()
