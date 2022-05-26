import glob
import random
import os
import numpy as np
import cv2

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class Datasets(Dataset):
    def __init__(self, hdr_image_dir, sdr_image_dir, image_size, train_txt, logtm):
        self.hdr_image_dir = hdr_image_dir
        self.sdr_image_dir = sdr_image_dir
        self.image_size = image_size

        # ----------
        #  Image paths
        # ----------
        self.hdr_image_paths = []
        self.sdr_image_paths = []


        if (train_txt is None):
            print("No exisiting text file for training images, reading from " + self.hdr_image_dir + " and " + self.sdr_image_dir)

            # for classification
            l_hdr_image_paths = []
            l_sdr_image_paths = []
            m_hdr_image_paths = []
            m_sdr_image_paths = []
            d_hdr_image_paths = []
            d_sdr_image_paths = []

            for HDRSDR in os.listdir(self.hdr_image_dir):  # for each video
                print (HDRSDR)

                # create four temp arrays, two for Mantiuk, two for Stelios
                man_hdr_image_paths = []
                ste_hdr_image_paths = []
                man_image_paths = []
                ste_image_paths = []

                dir_path_HDR=os. path. join(self.hdr_image_dir,str(HDRSDR))
                dir_path_SDR=os. path. join(self.sdr_image_dir,str(HDRSDR))
                if (os.path.isfile(dir_path_HDR)):
                    continue;
                for HDRfile in os.listdir(dir_path_HDR):
                    dir_path_HDR_file=os. path. join(dir_path_HDR,str(HDRfile))

                    for SDRfolders in os.listdir(dir_path_SDR): # for Mantiuk and Stelios
                        dir_path_SDR_TMO_folders=os. path. join(dir_path_SDR,str(SDRfolders))
                        if (os.path.isfile(dir_path_SDR_TMO_folders)):
                            continue;
                        for SDRfiles in os.listdir(dir_path_SDR_TMO_folders):
                            dir_path_SDR_files=os. path. join(dir_path_SDR_TMO_folders,str(SDRfiles))
                            if (HDRfile[:-4] in SDRfiles):
                                if ('Mantiuk' in dir_path_SDR_TMO_folders):
                                    man_hdr_image_paths.append(dir_path_HDR_file)
                                    man_image_paths.append(dir_path_SDR_files)
                                else:
                                    ste_hdr_image_paths.append(dir_path_HDR_file)
                                    ste_image_paths.append(dir_path_SDR_files)

                # classification
                man_dark_index = []
                ste_dark_index = []

                for i in range(len(man_hdr_image_paths)):
                    if (man_hdr_image_paths[i] in ste_hdr_image_paths):
                        # dark images
                        man_dark_index.append(i)
                    else:
                        # light images
                        l_hdr_image_paths.append(man_hdr_image_paths[i])
                        l_sdr_image_paths.append(man_image_paths[i])
                
                for i in range(len(ste_hdr_image_paths)):
                    if (ste_hdr_image_paths[i] in man_hdr_image_paths):
                        # dark images
                        ste_dark_index.append(i)
                    else:
                        # medium images
                        m_hdr_image_paths.append(ste_hdr_image_paths[i])
                        m_sdr_image_paths.append(ste_image_paths[i])

                for i in range(len(man_dark_index)):
                    # randomly select either Mantiuk or Stelios for dark scenes
                    rand = random.uniform(0, 1)
                    if (rand < 0.5):
                        d_hdr_image_paths.append(man_hdr_image_paths[man_dark_index[i]])
                        d_sdr_image_paths.append(man_image_paths[man_dark_index[i]])
                    else:
                        d_hdr_image_paths.append(ste_hdr_image_paths[ste_dark_index[i]])
                        d_sdr_image_paths.append(ste_image_paths[ste_dark_index[i]])

            len_list = [len(l_hdr_image_paths), len(m_hdr_image_paths), len(d_hdr_image_paths)]
            max_len = max(len_list)

            for i in range(max_len):
                if (i < len(l_hdr_image_paths)):
                    self.hdr_image_paths.append(l_hdr_image_paths[i])
                    self.sdr_image_paths.append(l_sdr_image_paths[i])
                if (i < len(m_hdr_image_paths)):
                    self.hdr_image_paths.append(m_hdr_image_paths[i])
                    self.sdr_image_paths.append(m_sdr_image_paths[i])
                if (i < len(d_hdr_image_paths)):
                    self.hdr_image_paths.append(d_hdr_image_paths[i])
                    self.sdr_image_paths.append(d_sdr_image_paths[i])

            # write to text file
            filename = str(logtm) + '_train.txt'
            print("Writing image paths to " + filename)

            j=0
            with open(filename, 'w') as f:
                for i in self.hdr_image_paths[:int(len(self.hdr_image_paths)*1)]:
                        f.write('%s\n' % i)
                        f.write('%s\n' % self.sdr_image_paths[j])
                        j=j+1
        else:
            print("Reading training image paths from " + train_txt)
            with open(train_txt, 'r') as f:
                lines = f.readlines()

                isHDR = True
                for l in lines:
                    if (isHDR):
                        self.hdr_image_paths.append(l.rstrip())
                    else:
                        self.sdr_image_paths.append(l.rstrip())

                    isHDR = not isHDR

        # ----------
        #  Transforms
        # ----------
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor()])

    def __getitem__(self, item):
        # ----------
        # Read HDR images
        # ----------
        hdr_file_name = self.hdr_image_paths[item] #.numpy().decode('utf-8')
        HDR = cv2.imread(hdr_file_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if ('.png' in hdr_file_name):
             HDR = HDR / 65536.0;

        HDR = torch.tensor(HDR).permute(2, 0, 1)

        # ----------
        # Read SDR images
        # ----------
        sdr_file_name = self.sdr_image_paths[item] #.numpy().decode('utf-8')
        SDR = cv2.imread(sdr_file_name, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        SDR = SDR / 255.0;

        SDR = torch.tensor(SDR).permute(2, 0, 1)

        # ----------
        # Apply transform (the same for both HDR and SDR)
        # ----------
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to transform
        torch.manual_seed(seed)
        HDR = self.transform(HDR)
            
        random.seed(seed) # apply this seed to transform
        torch.manual_seed(seed)
        SDR = self.transform(SDR)

        # rescale the pixel values to the -1 to 1 range
        SDR = SDR * 2.0 - 1.0

        images = {'hdr': HDR, 'sdr': SDR}
        return images

    def __len__(self):
        return len(self.hdr_image_paths)
