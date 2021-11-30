import os
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import copy
import random
import torch

class Data_train(data.Dataset):
    def __init__(self, config):
        self.batch_size = config.batch_size
        self.image_size = config.image_size
        self.image_path_celebA = config.image_path
        self.mask_path_celebA = config.mask_path
        
        self.imgs = []
        self.image_transforms1 = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
        ])
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1/255.,))
        ])
        f = open("CelebA-HQ-to-id-mapping.txt", mode='r')
        self.id_image_dict = {}
        for line in f.readlines()[0:25000]:
            items = line.split()
            name = items[0]
            id = items[2]
            if os.path.exists(os.path.join(self.mask_path_celebA, '{}.png'.format(name))):
                self.imgs.append((os.path.join(self.image_path_celebA, '{}.jpg'.format(name)),
                                os.path.join(self.mask_path_celebA, '{}.png'.format(name)),
                                id))
                if id in self.id_image_dict:
                    self.id_image_dict[id].append(os.path.join(self.image_path_celebA, '{}.jpg'.format(name)))
                else:
                    self.id_image_dict[id] = []
                    self.id_image_dict[id].append(os.path.join(self.image_path_celebA, '{}.jpg'.format(name)))

        self.imgs2 = copy.deepcopy(self.imgs)
        random.shuffle(self.imgs2)


    def __getitem__(self, index):
        train_images_filename, mask_images_filename, id =self.imgs[index]
        train_images_filename2, mask_images_filename2, id2 = self.imgs2[index]

        image_list = self.id_image_dict[id]
        index1 = random.randint(0, len(image_list)-1)
        image11_name = image_list[index1]

        image11 = Image.open(image11_name).convert('RGB')
        image22 = Image.open(train_images_filename2).convert('RGB')
        train_images = Image.open(train_images_filename).convert('RGB')
        mask_images_ori = Image.open(mask_images_filename)
        mask_images_ref = Image.open(mask_images_filename2)

        train_images = self.image_transforms1(train_images)
        train_images = np.array(train_images)
        mask_images_ori = np.array(mask_images_ori)

        train_images_fore = copy.deepcopy(train_images)
        train_images_bg = np.ones_like(train_images) * 255
        train_images_bg[mask_images_ori == 0] = train_images[mask_images_ori == 0]
 
        train_images = Image.fromarray(train_images)
        train_images_bg = Image.fromarray(train_images_bg)
        mask_images_ori = Image.fromarray(mask_images_ori)

        train_images = self.image_transforms(train_images)
        train_images_bg = self.image_transforms(train_images_bg)
        mask_images_ori = self.mask_transforms(mask_images_ori)
        mask_images_ref = self.mask_transforms(mask_images_ref)
        image11 = self.image_transforms(image11)
        image22 = self.image_transforms(image22)

        return train_images, train_images_bg, mask_images_ref, mask_images_ori, image11, image22

    def __len__(self):
        return len(self.imgs)
