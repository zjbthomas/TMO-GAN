import argparse
import os
import numpy as np
import math
import itertools
import sys
from sys import platform

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import lpips
import random

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--hdr_image_dir", type=str, default="/dataset/HDR/", help="name of the HDR dataset")
    parser.add_argument("--sdr_image_dir", type=str, default="/dataset/SDR", help="name of the SDR dataset")
    parser.add_argument("--train_images", type=str, default=None, help="text file with paths to training images")
    parser.add_argument("--image_size", type=int, default=120, help="size of the images")
    parser.add_argument("--batch_size", type=int, default=12, help="size of the batches")
    parser.add_argument("--lr", type=float, default=2e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--residual_blocks", type=int, default=23, help="number of residual blocks in the generator")
    parser.add_argument("--warmup_epochs", type=float, default=500, help="number of batches with pixel-wise loss only")
    parser.add_argument('--psnr_decay_epoch', type=int, default=50, help='psnr decay')
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay')
    parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
    parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")
    parser.add_argument("--run_name", type=str, default="C64_B4_E4_LRELU", help="run name")
    parser.add_argument("--log_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="batch interval between model checkpoints")
    parser.add_argument('--load_gen', type=str, help='pretrained gen')
    parser.add_argument('--load_dis', type=str, help='pretrained dis')
    parser.add_argument('--nGPU', type=int, default=1, help='number of gpus')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sdr_shape = (args.image_size, args.image_size)

    # Initialize generator and discriminator
    generator = GeneratorRRDB().to(device)
    discriminator = Discriminator(input_shape=(args.channels, *sdr_shape)).to(device)

    feature_extractor = FeatureExtractor().to(device)

    # Set feature extractor to inference mode
    feature_extractor.eval()

    # Losses
    criterion_GAN = nn.BCEWithLogitsLoss().to(device)
    criterion_content = nn.L1Loss().to(device)
    criterion_pixel = nn.L1Loss().to(device)
    criterion_lpips_vgg = lpips.LPIPS().to(device)
    criterion_psnr = nn.MSELoss().to(device)

    # Load pretrained models
    if args.load_gen != None:
        print('Load pretrained generator: ' + args.load_gen)
        generator.load_state_dict(torch.load(args.load_gen))
    if args.load_dis != None:
        print('Load pretrained discriminator: ' + args.load_dis)
        discriminator.load_state_dict(torch.load(args.load_dis))

    # output model details
    summary(generator, (args.channels, *sdr_shape))
    summary(discriminator, (args.channels, *sdr_shape))

    # Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # Time for log
    logtm = datetime.now().strftime("%Y%m%d%H%M%S")

    # Dataset
    train_txt = None
    if (args.train_images is not None):
        train_txt = args.train_images

    train_dataset = Datasets(args.hdr_image_dir, args.sdr_image_dir, args.image_size, train_txt, logtm)
    dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True, drop_last=True)

    # Conversion from epoch to step/iter
    psnr_decay_iter = args.psnr_decay_epoch * len(dataloader)
    decay_iter = args.decay_epoch * len(dataloader)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    lr_scheduler_generator_psnr = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_G,
                                                                step_size=psnr_decay_iter,
                                                                gamma=0.5)
    lr_scheduler_generator = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_G,
                                                                step_size=decay_iter,
                                                                gamma=0.5)

    lr_D = args.lr * (0.5 ** (args.warmup_epochs / args.psnr_decay_epoch - 1))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_D, betas=(args.b1, args.b2))
    
    lr_scheduler_discriminator = torch.optim.lr_scheduler.StepLR(optimizer=optimizer_D,
                                                                    step_size=decay_iter,
                                                                    gamma=0.5)

    # ----------
    #  Training
    # ----------
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter("logs/" + logtm + "_" + args.run_name)
    checkpoint_dir = "checkpoints/" + logtm + "_" + args.run_name
    os.makedirs(checkpoint_dir, exist_ok=True)

    # loss sum for step
    step_total_psnr = 0
    step_total_lpips = 0

    random.seed(args.seed)

    for epoch in range(args.epoch, args.n_epochs):
        print("Starting Epoch ", epoch + 1)

        # loss sum for epoch
        epoch_total_content = 0
        epoch_total_adv = 0
        epoch_total_pixel = 0

        epoch_total_G = 0
        epoch_total_D = 0

        epoch_total_psnr = 0
        epoch_total_lpips = 0

        epoch_steps = 0

        for step, imgs in enumerate(dataloader):
            curr_steps = epoch * len(dataloader) + step + 1

            # Configure model input
            imgs_hdr = Variable(imgs["hdr"].type(Tensor))
            imgs_sdr = Variable(imgs["sdr"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((imgs_hdr.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((imgs_hdr.size(0), *discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # Generate an SDR image from HDR input
            gen_sdr = generator(imgs_hdr)

            # Measure pixel-wise loss against ground truth
            loss_pixel = criterion_pixel(gen_sdr, imgs_sdr)

            psnr = 10 * torch.log10(1.0 / criterion_psnr(imgs_sdr, gen_sdr)).item()
            step_total_psnr += psnr
            step_avg_psnr = step_total_psnr / curr_steps

            loss_lpips = torch.mean(criterion_lpips_vgg.forward(imgs_sdr, gen_sdr, normalize=True)).item()
            step_total_lpips += loss_lpips
            step_avg_lpips = step_total_lpips / curr_steps

            if curr_steps < args.warmup_epochs * len(dataloader):
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                lr_scheduler_generator_psnr.step()
                if step % args.log_interval == 0:
                    print(f"Warmup: [Epoch {epoch + 1}/{args.n_epochs}] [Batch {step + 1}/{len(dataloader)}] "
                        f"[Pixel Loss {loss_pixel.item():.3e}]"
                        f"[Avg PSNR {step_avg_psnr:.3f}]"
                        f"[Avg LPIPS {step_avg_lpips:.3f}]"
                        f"")
                    writer.add_scalar("LearningRate/Generator Warmup", lr_scheduler_generator_psnr.get_last_lr()[0], curr_steps)
                    writer.add_scalar("Metrics/AVG PSNR", step_avg_psnr, epoch * len(dataloader) + step)
                    writer.add_scalar("Metrics/AVG LPIPS", step_avg_lpips, epoch * len(dataloader) + step)
                    writer.add_images('HDR', imgs_hdr, epoch * len(dataloader) + step)

                    clamp_imgs_sdr = imgs_sdr.clone().detach()
                    clamp_imgs_sdr = (clamp_imgs_sdr + 1.0) / 2.0
                    writer.add_images('SDR', clamp_imgs_sdr, epoch * len(dataloader) + step)

                    clamp_gen_sdr = gen_sdr.clone().detach()
                    clamp_gen_sdr = (clamp_gen_sdr + 1.0) / 2.0
                    writer.add_images('Gen SDR', clamp_gen_sdr, epoch * len(dataloader) + step)
                if step % args.checkpoint_interval == 0:
                    tm = datetime.now().strftime("%Y%m%d%H%M%S")
                    torch.save(generator.state_dict(),
                            os.path.join(checkpoint_dir, tm + '_' + args.run_name + '_gen_' + str(
                                epoch + 1) + "_" + str(step + 1) + '.pth'))
                continue

            # Extract validity predictions from discriminator
            pred_real = discriminator(imgs_sdr).detach()
            pred_fake = discriminator(gen_sdr)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            # Content loss
            gen_features = feature_extractor(gen_sdr)
            real_features = feature_extractor(imgs_sdr).detach()
            loss_content = criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = loss_content + args.lambda_adv * loss_GAN + args.lambda_pixel * loss_pixel

            loss_G.backward()
            optimizer_G.step()

            # log losses for epoch
            epoch_steps += 1

            epoch_total_content += loss_content
            epoch_total_adv += loss_GAN
            epoch_total_pixel += loss_pixel
            epoch_total_G += loss_G

            epoch_total_psnr += psnr
            epoch_total_lpips += loss_lpips
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            pred_real = discriminator(imgs_sdr)
            pred_fake = discriminator(gen_sdr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2

            # log losses for epoch
            epoch_total_D += loss_D

            loss_D.backward()
            optimizer_D.step()
            lr_scheduler_generator.step()
            lr_scheduler_discriminator.step()
            # --------------
            #  Log Progress (for step)
            # --------------
            if step % args.log_interval == 0:
                print(f"[Epoch {epoch + 1}/{args.n_epochs}] [Batch {step + 1}/{len(dataloader)}] "
                    f"[D Loss {loss_D.item():.3f}]"
                    f"[G Loss {loss_G.item():.3f}]"
                    f"[Pixel Loss {args.lambda_pixel * loss_pixel.item():.3e}]"
                    f"[Adv Loss {args.lambda_adv * loss_GAN.item():.3e}]"
                    f"[Content Loss {loss_content.item():.3e}]"
                    f"[Avg PSNR {step_avg_psnr:.3f}]"
                    f"[Avg LPIPS {step_avg_lpips:.3f}]"
                    f"")

                writer.add_scalar("LearningRate/Generator", lr_scheduler_generator.get_last_lr()[0],
                                curr_steps)
                writer.add_scalar("LearningRate/Discriminator", lr_scheduler_discriminator.get_last_lr()[0],
                                curr_steps)
                writer.add_scalar("Loss/Total D Loss", loss_D.item(), epoch * len(dataloader) + step)
                writer.add_scalar("Loss/Content", loss_content.item(), curr_steps)
                writer.add_scalar("Loss/Adversarial", args.lambda_adv * loss_GAN.item(), curr_steps)
                writer.add_scalar("Loss/Perceptual", args.lambda_pixel * loss_pixel.item(), curr_steps)
                writer.add_scalar("Loss/Total G Loss", loss_G.item(), curr_steps)
                writer.add_scalar("Metrics/AVG PSNR", step_avg_psnr, epoch * len(dataloader) + step)
                writer.add_scalar("Metrics/AVG LPIPS", step_avg_lpips, epoch * len(dataloader) + step)
                writer.add_images('HDR', imgs_hdr, epoch * len(dataloader) + step)

                clamp_imgs_sdr = imgs_sdr.clone().detach()
                clamp_imgs_sdr = (clamp_imgs_sdr + 1.0) / 2.0
                writer.add_images('SDR', clamp_imgs_sdr, epoch * len(dataloader) + step)

                clamp_gen_sdr = gen_sdr.clone().detach()
                clamp_gen_sdr = (clamp_gen_sdr + 1.0) / 2.0
                writer.add_images('Gen SDR', clamp_gen_sdr, epoch * len(dataloader) + step)

            if step % args.checkpoint_interval == 0:
                tm = datetime.now().strftime("%Y%m%d%H%M%S")
                torch.save(generator.state_dict(),
                        os.path.join(checkpoint_dir, tm + '_' + args.run_name + '_gen_' + str(
                            epoch + 1) + "_" + str(step + 1) + '.pth'))
                torch.save(discriminator.state_dict(),
                        os.path.join(checkpoint_dir, tm + '_' + args.run_name + '_dis_' + str(
                            epoch + 1) + "_" + str(step + 1) + '.pth'))

        # --------------
        #  Log Progress (for epoch)
        # --------------
        # loss average for epoch
        if (epoch_steps != 0):
            epoch_avg_content = epoch_total_content / epoch_steps
            epoch_avg_adv = epoch_total_adv / epoch_steps
            epoch_avg_pixel = epoch_total_pixel / epoch_steps
            epoch_avg_G = epoch_total_G / epoch_steps
            epoch_avg_D = epoch_total_D / epoch_steps
            epoch_avg_psnr = epoch_total_psnr / epoch_steps
            epoch_avg_lpips = epoch_total_lpips / epoch_steps

            print(f"[Epoch {epoch + 1}/{args.n_epochs}]"
                    f"[Epoch D Loss {epoch_avg_D:.3f}]"
                    f"[Epoch G Loss {epoch_avg_G:.3f}]"
                    f"[Epoch Pixel Loss {args.lambda_pixel * epoch_avg_pixel:.3e}]"
                    f"[Epoch Adv Loss {args.lambda_adv * epoch_avg_adv:.3e}]"
                    f"[Epoch Content Loss {epoch_avg_content:.3e}]"
                    f"[Epoch Avg PSNR {epoch_avg_psnr:.3f}]"
                    f"[Epoch Avg LPIPS {epoch_avg_lpips:.3f}]"
                    f"")

            writer.add_scalar("Epoch LearningRate/Generator", lr_scheduler_generator.get_last_lr()[0],
                                epoch)
            writer.add_scalar("Epoch LearningRate/Discriminator", lr_scheduler_discriminator.get_last_lr()[0],
                                epoch)
            writer.add_scalar("Epoch Loss/Total D Loss", epoch_avg_D, epoch)
            writer.add_scalar("Epoch Loss/Content", epoch_avg_content, epoch)
            writer.add_scalar("Epoch Loss/Adversarial", args.lambda_adv * epoch_avg_adv, epoch)
            writer.add_scalar("Epoch Loss/Perceptual", args.lambda_pixel * epoch_avg_pixel, epoch)
            writer.add_scalar("Epoch Loss/Total G Loss", epoch_avg_G, epoch)
            writer.add_scalar("Epoch Metrics/AVG PSNR", epoch_avg_psnr, epoch)
            writer.add_scalar("Epoch Metrics/AVG LPIPS", epoch_avg_lpips, epoch)
            writer.add_images('Epoch HDR', imgs_hdr, epoch)

            clamp_imgs_sdr = imgs_sdr.clone().detach()
            clamp_imgs_sdr = (clamp_imgs_sdr + 1.0) / 2.0
            writer.add_images('Epoch SDR', clamp_imgs_sdr, epoch)

            clamp_gen_sdr = gen_sdr.clone().detach()
            clamp_gen_sdr = (clamp_gen_sdr + 1.0) / 2.0
            writer.add_images('Epoch Gen SDR', clamp_gen_sdr, epoch)
    print("Finished")

if platform == "win32":
    if __name__ == '__main__':
        main()
else:
    main()