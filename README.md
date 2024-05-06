# Cross-Domain Generalization

Temporary PyTorch implementation of [Cross-Domain Generalization: Enhancing Rare Disease Data
Representation using Diffusion Model](https://prbs5kong.github.io/assets/pdf/CVPRW_01.pdf) by [Wonseok Oh](https://prbs5kong.github.io/) et al.

<p align="center">
  <img src="https://github.com/prbs5kong/crossdomain_gen/blob/main/assets/1.png" />
</p>


We propose the **Cross-Domain Genralization model**, which translates the input domain data type A into output domain data B.

## Environment
```
$ conda create -n cd_gen python=3.6
$ pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
$ conda install -c conda-forge packaging 
$ conda install -c "conda-forge/label/cf201901" visdom 
$ conda install -c conda-forge gputil 
$ conda install -c conda-forge dominate 
```

## Dataset Download
Download the dataset with following script e.g.

```
bash ./datasets/download_cut_dataset.sh (update required)
```

please refer to the original repository of [CUT](https://github.com/taesungp/contrastive-unpaired-translation) and [UNSB](https://github.com/cyclomon/UNSB)

## Training 
Refer the ```./run_train.sh``` file or

```
python train.py --dataroot ./datasets/Domain_A_penumonia --name A2B \
--mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 -direction A2B
```

for reverse direction run the below code, 

```
python train.py --dataroot ./datasets/Domain_A_penumonia --name B2A \
--mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 -direction B2A
```

Although the training is available with arbitrary batch size, we recommend to use batch size = 1.

## Test & Evaluation
Refer the ```./run_test.sh``` file or

```
python test.py --dataroot [path/to/dataset] --name [experiment] --mode [user-mode] \
--phase test --epoch [test-epoch] --eval --num_test [number-image] \
--gpu_ids 0 --checkpoints_dir ./checkpoints
```

The outputs will be saved in ```./results/[experiment]/```

Folders named as ```fake_[num_NFE]``` represent the generated outputs with different NFE steps.

For evaluation, we use official module of [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

```
python -m pytorch_fid [/output/path] [/real/path]
```

```/real/path``` should be test images of target domain. 

For testing on our vgg-based trained model, 

Refer the ```./vgg_sb/scripts/test_sc_main.sh``` file 

The pre-trained checkpoints are provided [here](https://drive.google.com/drive/folders/1Q8tuBGegMMHd9PzvcklDm0wM1sE4PPwK?usp=sharing)

## References

```
Will be updated
```

### Acknowledgement
Our source code is based on [CUT](https://github.com/taesungp/contrastive-unpaired-translation) and [UNSB](https://github.com/cyclomon/UNSB). \
We used [pytorch-fid](https://github.com/mseitzer/pytorch-fid) for FID calculation. \
We modified the network based on the implementation of [DDGAN](https://github.com/NVlabs/denoising-diffusion-gan).
