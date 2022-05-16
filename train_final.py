# -----------------------------------------------------------
# Load swav pretrained backbone and train FasterRCNN using supervised dataset
#
#   note: only work for resnet50
# -----------------------------------------------------------
import argparse
import build_model
import copy
import customized_module as my_nn
from dataset import UnlabeledDataset, LabeledDataset
from engine import train_one_epoch, evaluate
import os
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import swav_resnet as resnet_models
import sys
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
import utils

parser = argparse.ArgumentParser(description="Evaluate models: Fine-tuning with labeled dataset")
#########################
#### swav parameters ####
#########################
parser.add_argument("--pretrained_hub", default=0, type=int,
                    help="if backbone downloaded from Facebook hub")
parser.add_argument("--swav_file", type=str, default="swav_res50_ep100.pth",
                    help="path to swav checkpoints")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=2000, type=int,
                    help="number of prototypes")

#############################
#### training parameters ####
#############################
parser.add_argument("--mode", choices=['train', 'eval', 'resume'], 
                    default='train', type=str, help="Choose action.") #TODO: add resume impl
parser.add_argument("--epochs", default=40, type=int,
                    help="number of total epochs to run")
parser.add_argument("--eval_freq", default=2, type=int,
                    help="Eval the model periodically")
parser.add_argument("--checkpoint_freq", type=int, default=2,
                    help="Save the model periodically")
parser.add_argument("--arch", choices=['resnet50', 'resnet18', 'resnet34'],
                    default='resnet50', type=str, help="Architecture")
parser.add_argument("--high_lr", default=5e-3, type=float,
                    help="lr for rcnn")
parser.add_argument("--low_lr", default=5e-5, type=float,
                    help="lr for transfer layer in backbone")
parser.add_argument("--super_low_lr", default=5e-7, type=float,
                    help="lr for first block in backbone")
parser.add_argument("--sched_step", default=6, type=int,
                    help="Step size of lr scheduler")
parser.add_argument("--sched_gamma", default=0.2, type=float,
                    help="lr scheduler")
parser.add_argument("--norm_layer", choices=['FBN', 'GN', 'IN', 'BN'],
                    default='FBN', type=str, help="Norm layer type.")
parser.add_argument("--box_head", choices=['mlp', 'res'],
                    default='mlp', type=str, help="box head type.")
##########################
#### other parameters ####
##########################
parser.add_argument("--workers", default=2, type=int,
                    help="number of data loading workers")
parser.add_argument("--data_path", type=str, default="../../data/labeled",
                    help="path to imagenet")
parser.add_argument("--checkpoint_path", type=str, default="./back_up",
                    help="path to save back up model's weights")
parser.add_argument("--checkpoint_file", type=str, default="",
                    help="name of saved model")


parser.add_argument("--debug", default=0, type=int,
                    help="DEBUG architecture of model")
parser.add_argument("--gcp_sucks", default=1, type=int,
                    help="you know the answer")

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def load_pretrained_swav(ckp_path):
    if (args.arch == "resnet18"):
        template = resnet_models.resnet18
    elif (args.arch == "resnet34"):
        template = resnet_models.resnet34
    elif (args.arch == "resnet50"):
        template = resnet_models.resnet50
    else:
        print("??")
        raise Exception
    model = template(
        normalize=True,
        hidden_mlp=args.hidden_mlp, 
        output_dim=args.feat_dim, 
        nmb_prototypes=args.nmb_prototypes
    )    
    if os.path.isfile(ckp_path):
        state_dict = torch.load(ckp_path, map_location="cuda")
        
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # remove prefixe "module."
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        for k, v in model.state_dict().items():
            
            if k not in state_dict:
                print('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                print('key "{}" is of different shape in model and provided state dict'.format(k))
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print("Load pretrained model with msg: {}".format(msg))
    else:
        raise Exception("No pretrained weights found")
    return model


def get_model(num_classes, returned_layers=None):
    if args.mode == 'eval' or args.mode == 'resume':
        model = torch.load(os.path.join(args.checkpoint_path, args.checkpoint_file))
        model.eval()
        return model

    if args.pretrained_hub == 1:
        backbone = torch.hub.load("facebookresearch/swav", "resnet50")
        print("Load pretrained swav backbone from Facebook hub")
    else:
        backbone = load_pretrained_swav(args.swav_file)
        backbone.padding = nn.Identity() #TODO: need to double check this
        backbone.projection_head = nn.Identity()
        backbone.prototypes = nn.Identity()
        print("Load pretrained swav backbone from OUR checkpoint")
    backbone.avgpool = nn.Identity()
    backbone.fc = nn.Identity()
    # --------------------Map resnet backbone---------------------
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]
    if min(returned_layers) <= 0 or max(returned_layers) >= 5:
        raise ValueError(f"Each returned layer should be in the range [1,4]. Got {returned_layers}")
    return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}

    new_back_bone = my_nn.IntermediateLayerGetter(backbone, return_layers=return_layers)

    # --------------------standard demo thing---------------------

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False)

    # Add normalize layer based on labeled dataset
    min_size, max_size = 800, 1333
    image_mean, image_std = [0.4697, 0.4517, 0.3954], [0.2305, 0.2258, 0.2261]
    norm_layer = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    model.transform = norm_layer

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # ---------------------------------------------------------
    # Change backbone
    # ---------------------------------------------------------

    new_back_bone_copy = copy.deepcopy(new_back_bone)
    model.backbone.body = new_back_bone
    if args.norm_layer == "FBN":
        build_model.replace_bn_FBN(model.backbone.body, "backbone.body")
        model.backbone.body.load_state_dict(new_back_bone_copy.state_dict())
    elif args.norm_layer == "GN":
        build_model.replace_bn_GN(model.backbone.body, "backbone.body")
    elif args.norm_layer == "IN":
        build_model.replace_bn_IN(model.backbone.body, "backbone.body")
    elif args.norm_layer == "BN":
        # build_model.replace_bn_IN(model.backbone.body, "backbone.body")
        pass
    else:
        print("WARNING: Norm Layer not replaced")

    

    for m in model.parameters():
        m.requires_grad = True

    # ---------------------------------------------------------
    # Change fpn
    # ---------------------------------------------------------
    # ResNet-18
    if args.arch == "resnet18" or args.arch == "resnet34":
        model.backbone.fpn.inner_blocks[0] = nn.Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1))
        model.backbone.fpn.inner_blocks[1] = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        model.backbone.fpn.inner_blocks[2] = nn.Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
        model.backbone.fpn.inner_blocks[3] = nn.Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1))

    # ---------------------------------------------------------
    # Change box_head
    # ---------------------------------------------------------
    if args.box_head == "mlp":
        out_channels = model.backbone.out_channels
        resolution = model.roi_heads.box_roi_pool.output_size[0]
        representation_size = 1024
        new_box_head = my_nn.CustomizedBoxHead(out_channels * resolution ** 2, representation_size)
    elif args.box_head == "res":
        new_box_head = my_nn.ResBoxHead()
    model.roi_heads.box_head = new_box_head

    if args.debug == 1:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f'TRAIN: {name}')
            else:
                print(f'FROZEN: {name}')

        for name, child in model.named_children():
            print(f'name is: {name}')
            print(f'module is: {child}')
        print(model.backbone.body.layer4[0].bn1.weight)
        print(model.backbone.body.layer4[0].bn1.weight)
        print("DONE")
        sys.stdout.flush()
    return model

def main():
    global args
    args = parser.parse_args()

    if args.gcp_sucks == 1:
        args.data_path = '/labeled'

    if args.debug == 1:
        args.eval_freq = 420
        args.checkpoint_freq = 420
        args.epochs = 30
        args.sched_step = 10

    print('--------------------Args--------------------')
    print(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))
    print('--------------------------------------------')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if not os.path.isdir(args.checkpoint_path):
        os.mkdir(args.checkpoint_path)

    if args.gcp_sucks == 0:
        pd_train_header = [
            "Epoch", "loss", "loss_classifier", 
            "loss_box_reg", "loss_objectness",
            "loss_rpn_box_reg",
        ]
        filename = "train_stats_test.pkl" if (args.debug == 1) else "train_stats_detailed.pkl"
        training_stats_detailed = utils.PD_Stats(
            os.path.join(args.checkpoint_path, filename), 
            pd_train_header,
        )

        training_stats = utils.PD_Stats(
            os.path.join(args.checkpoint_path, "train_stats.pkl"), 
            ["loss"],
        )
    
    num_classes = 100
    train_dataset = LabeledDataset(root=args.data_path, split="training", transforms=get_transform(train=True))
    valid_dataset = LabeledDataset(root=args.data_path, split="validation", transforms=get_transform(train=False))
    
    if args.debug == 1:
        index = list(range(10))
        train_dataset = torch.utils.data.Subset(train_dataset, index)
        valid_dataset = torch.utils.data.Subset(valid_dataset, index)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, collate_fn=utils.collate_fn)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=2, collate_fn=utils.collate_fn)
    
    model = get_model(num_classes)
    model.to(device)
    if args.gcp_sucks == 1:
        print(model)

    super_low_param = []
    super_low_name = []
    low_lr_param = []
    low_name = []
    high_lr_param = []
    high_name = []
    for name, param in model.named_parameters():
        if "backbone.body" in name:
            if "bn" in name:
                if args.norm_layer == "FBN":
                    pass
                    # print(f'reset running stats for {name}')
                    # param.running_var.fill_(1)
                    # param.running_mean.zero_()
                elif args.norm_layer == "BN":
                    if "layer1" in name or "backbone.body.bn" in name:
                        super_low_param.append(param)
                        super_low_name.append(name)
                    else:
                        low_lr_param.append(param)
                        low_name.append(name)
                else:
                    low_lr_param.append(param)
                    low_name.append(name)
            elif "layer1" in name or "backbone.body.conv1" in name:
                super_low_param.append(param)
                super_low_name.append(name)
            else:
                low_lr_param.append(param)
                low_name.append(name)
        else:
            high_lr_param.append(param)
            high_name.append(name)

    optimizer = torch.optim.SGD(
         [
            {"params": high_lr_param},
            {"params": low_lr_param, "lr": args.low_lr},
            {"params": super_low_param, "lr": args.super_low_lr},
         ],
         lr=args.high_lr,
         momentum=0.9,
         weight_decay=0.0005
    )
    print(f'{high_name=}')
    print(f'{low_name=}')
    print(f'{super_low_name=}')
    for param_group in optimizer.param_groups:
        print(param_group['lr'])
    # raise NotImplementedError
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.sched_step, gamma=args.sched_gamma)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4, 8, 16], gamma=0.1)
    # optionally resume from a checkpoint
    to_restore = {"epoch": 0}
    if args.mode == 'resume':
        utils.restart_from_checkpoint(
            os.path.join(args.checkpoint_path, "checkpoint.pth.tar"),
            run_variables=to_restore,
            state_dict=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
    start_epoch = to_restore["epoch"]

    eval_result = {}

    if (args.mode == "train") or (args.mode == "resume"):
        for epoch in range(start_epoch, args.epochs):
            # train for one epoch, printing every 10 iterations
            train_log, smooth_loss_hist = train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=5000)
            
            if args.gcp_sucks == 0:
                training_stats_detailed.update(train_log)
                training_stats.update_col(smooth_loss_hist)

            save_dict = {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                'scheduler_state': lr_scheduler.state_dict()
            }
            torch.save(
                save_dict,
                os.path.join(args.checkpoint_path, "checkpoint.pth.tar"),
            )
            # update the learning rate
            lr_scheduler.step()


            if args.debug == 0:
                # save whole model instead of state_dict
                if (((epoch + 1) % args.checkpoint_freq == 0 and (epoch > 0)) or (epoch == args.epochs - 1)):
                    torch.save(model, os.path.join(args.checkpoint_path, f"model_{epoch}.pth"))

                if (((epoch % args.eval_freq) == 0) or (epoch == args.epochs - 1)):
                    # evaluate on the test dataset
                    coco_res, _ = evaluate(model, valid_loader, device=device)
                    # print(f'coco_res: {coco_res.coco_eval["bbox"].stats}')
                    eval_result[epoch] = coco_res.coco_eval
                    

    # fuck GCP
    # with open(os.path.join(args.checkpoint_path, "eval_res.pickle"), "w") as outfile:
    #     pickle.dump(eval_result, outfile)
    torch.save(model, "model_final.pth")
if __name__ == "__main__":
    main()