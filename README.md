# A Deep Learning Based Tone Mapping Operator for 4K HDR Images Focusing on Color Accuracy and Artistic Intent

This is the repository for paper [A Deep Learning Based Tone Mapping Operator for 4K HDR Images Focusing on Color Accuracy and Artistic Intent]() submitted to ICIP 2022.

## Dataset

The dataset we constructed in Section 2.1 of our paper can be downloaded [here]().

## Experiments

### Training

Run the following code to train the network:

```
python -u esrgan.py --hdr_image_dir /path/to/HDR/ --sdr_image_dir /path/to/SDR/ --residual_blocks 23 --lr 1e-5 --warmup_epochs 300 --image_size 128 --batch_size 8 --psnr_decay_epoch 50 --decay_epoch 100 --n_epochs 600
```

### Testing

Run the following code to evaluate the network. A pretrained model is provided [here](./pretrained/model.pth).

```
python -u infer.py --image_path /path/to/HDR/dir/ 
--output_dir /path/to/output/dir/ --checkpoint_model ./pretrained/model.pth 
```