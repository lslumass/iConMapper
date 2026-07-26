DSGPM-TP is a deep learning-based model designed to automate the process of coarse-grained (CG) mapping from fine-grained molecular structures. By utilizing graph-based neural networks, the model partitions atomic structures into CG representations and predicts the types of CG particles. This tool reduces manual effort and enhances reproducibility and accuracy in CG mapping. More details can be accessed [here](https://doi.org/10.48550/arXiv.2408.06609).


## Model Training

To train the DSGPM-TP model on your dataset, follow these steps:

###  1. Prepare Your Dataset

Each molecule item (`.json`) should include node features, edges, and the target CG labels.
The `.json` example can be found in our prebuilt dataset of MARTINI2 `MARTINI2_Dataset`.

###  2. Train Your Model

You can use the following Shell script to train the DSGPM-TP model.
```
cgloss=0.01
mkdir ./ckpt/${cgloss}
mkdir ./tb_log/${cgloss}

CUDA_VISIBLE_DEVICES=0 python ./train.py --data_root ./MARTINI_Dataset --epoch 500 --batch_size 32 --ckpt ./ckpt/${cgloss}  --num_workers 4 --tb_root  ./tb_log/${cgloss} --tb_log --title cgloss_${cgloss} --cg_type_loss_parameter ${cgloss} --no_charge_feat --no_aromatic_feat 
```
