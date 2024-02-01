# Obtained from: https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License
# A copy of the license is available at resources/license_dacs

import kornia
import numpy as np
import torch
import torch.nn as nn


def strong_transform(param, data=None, target=None):
    assert ((data is not None) or (target is not None))
    data, target = one_mix(mask=param['mix'], data=data, target=target)
    data, target = color_jitter(
        color_jitter=param['color_jitter'],
        s=param['color_jitter_s'],
        p=param['color_jitter_p'],
        mean=param['mean'],
        std=param['std'],
        data=data,
        target=target)
    data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
    return data, target


def get_mean_std(img_metas, dev):
    mean = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['mean'], device=dev)
        for i in range(len(img_metas))
    ]
    mean = torch.stack(mean).view(-1, 3, 1, 1)
    std = [
        torch.as_tensor(img_metas[i]['img_norm_cfg']['std'], device=dev)
        for i in range(len(img_metas))
    ]
    std = torch.stack(std).view(-1, 3, 1, 1)
    return mean, std


def denorm(img, mean, std):
    return img.mul(std).add(mean) / 255.0


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter > p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def get_class_masks(labels):
    class_masks = []
    for label in labels:
        classes = torch.unique(labels)
        nclasses = classes.shape[0]
        class_choice = np.random.choice(
            nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
        classes = classes[torch.Tensor(class_choice).long()]
        class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
    return class_masks


def generate_class_mask(label, classes):
    label, classes = torch.broadcast_tensors(label,
                                             classes.unsqueeze(1).unsqueeze(2))
    class_mask = label.eq(classes).sum(0, keepdims=True)
    return class_mask


def one_mix(mask, data=None, target=None):
    if mask is None:
        return data, target
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0])
        data = (stackedMask0 * data[0] +
                (1 - stackedMask0) * data[1]).unsqueeze(0)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0 * target[0] +
                  (1 - stackedMask0) * target[1]).unsqueeze(0)
    return data, target

from PIL import Image
def scale_jittering(images, labels, weights, means, stds, dev):
    random_scale = 0.1 + 0.9 * np.random.rand() # random value between 0.1 and 1.0

    new_images = []
    new_labels = []
    new_weights = []

    for i, img in enumerate(images):
        print(img.shape)
        print(labels[i].shape)
        print(weights[i].shape)
        img = tensor_to_image(img[0], means, stds)
        if (1.0 - random_scale)*img.width < 1.0 or (1.0 - random_scale)*img.height < 1.0:
            pos = (0,0)
        else:
            pos = (np.random.randint((1.0 - random_scale)*img.width), np.random.randint((1.0 - random_scale)*img.height))

        temp_img = Image.new('RGB', img.size)
        rescaled_img = img.resize((int(random_scale * img.size[0]), int(random_scale * img.size[1])), resample=Image.BILINEAR)
        temp_img.paste(rescaled_img, pos)
        temp_img = torch.tensor(np.array(temp_img).transpose((2,0,1)), device=dev).float()
        temp_img = temp_img / 255.0
        renorm_(temp_img, means[0,:,:,:], stds[0,:,:,:])
        temp_img = temp_img.unsqueeze(0)
        print("temp img shape", temp_img.shape)
        new_images.append(temp_img)

        if not labels is None:
            label = labels[i][0,0,:,:].detach().cpu().numpy()
            print(label.shape)
            label = label.astype(np.uint8)
            print(label)
            label = Image.fromarray(label)
            temp_label = Image.new('L', label.size, color=255)
            rescaled_label = label.resize((int(random_scale * label.size[0]), int(random_scale * label.size[1])), resample=Image.NEAREST)
            temp_label.paste(rescaled_label, pos)
            temp_label = torch.tensor(np.array(temp_label), device=dev).unsqueeze(0).unsqueeze(0).long()
            print("temp label shape", temp_label.shape)
            new_labels.append(temp_label)

        if not weights is None:
            weight = weights[i]
            weight = Image.fromarray(weight.detach().cpu().numpy().astype(np.uint8))
            temp_weight = Image.new('L', weight.size, color=0)
            rescaled_weight = weight.resize((int(random_scale * weight.size[0]), int(random_scale * weight.size[1])), resample=Image.NEAREST)
            temp_weight.paste(rescaled_weight, pos)
            temp_weight = torch.tensor(np.array(temp_weight), device=dev).unsqueeze(0)
            print("temp weight shape", temp_weight.shape)
            new_weights.append(temp_weight)

    return new_images, new_labels, new_weights

def tensor_to_image(tensor, mean, std):
    img = torch.clamp(denorm(tensor, mean[0,:,:,:], std[0,:,:,:]), 0, 1)
    temp = img.detach().cpu().numpy().transpose((1,2,0))
    temp = 255 * temp
    temp = temp.astype(np.uint8)

    return Image.fromarray(temp)
