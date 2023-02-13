SINGLE NODE SINGLE GPU

To run:

`python -u main_swav.py --epochs 100 --queue_length 3840 --batch_size 128 --base_lr 0.01 --final_lr 0.00001 --warmup_epochs 10 --start_warmup 0.3 --arch resnet18 --freeze_prototypes_niters 5000 --checkpoint_freq 3 --data_path ../unlabeled_data --nmb_prototypes 1000`