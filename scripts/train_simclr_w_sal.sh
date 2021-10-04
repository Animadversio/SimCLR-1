#!/bin/bash
#BSUB -n 8
#BSUB -q general
#BSUB -G compute-crponce
#BSUB -J 'simclr_fast_sal_train[1-4]'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB"
#BSUB -R 'gpuhost'
#BSUB -R 'select[mem>64G]'
#BSUB -R 'rusage[mem=64GB]'
#BSUB -M 64GB
#BSUB -u binxu.wang@wustl.edu
#BSUB -o  /scratch1/fs1/crponce/simclr_fast_salienc_train.%J.%I
#BSUB -a 'docker(pytorchlightning/pytorch_lightning:base-cuda-py3.9-torch1.9)'

echo "$LSB_JOBINDEX"

param_list=\
'--out_dim 128 --batch-size 256 --run_label proj128_eval_sal_T3.0_pad --crop_temperature 3.0
--out_dim 128 --batch-size 256 --run_label proj128_eval_sal_T3.0_pad --crop_temperature 3.0 --pad_img False
--out_dim 128 --batch-size 256 --run_label proj128_eval_sal_T0.2_pad --crop_temperature 0.2
--out_dim 128 --batch-size 256 --run_label proj128_eval_sal_T0.2_pad --crop_temperature 0.2 --pad_img False
'

export extra_param="$(echo "$param_list" | head -n $LSB_JOBINDEX | tail -1)"
echo "$extra_param"

cd ~/SimCLR-torch/
python run_salcrop.py -data $SCRATCH1/Datasets -dataset-name stl10 --workers 16 --ckpt_every_n_epocs 5 --epochs 100  $extra_param 