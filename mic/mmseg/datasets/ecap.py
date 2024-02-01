# ---------------------------------------------------------------
# Copyright (c) 2023-2024 Volvo Group, Erik Brorsson. All rights reserved.
# Licensed under the MIT License
# ---------------------------------------------------------------

from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset
import numpy as np
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # not sure what this does
import time
import torch 
import json
import os
import queue
from typing import List
from sortedcontainers import SortedList
from threading import Lock, Thread, active_count
import mmcv
import shutil
import os.path as osp


def denorm(img, mean, std):
    """Denormalize and image."""
    return img.mul(std).add(mean) / 255.0

def tensor_to_image(tensor, mean, std):
    """Convert a torch.Tensor to an Image.Image."""
    assert isinstance(tensor, torch.Tensor), f"input is not a tensor, but rather {type(tensor)}"
    mean = mean.to(device=tensor.device)
    std = std.to(device=tensor.device)
    img = torch.clamp(denorm(tensor, mean[0,:,:,:], std[0,:,:,:]), 0, 1)
    temp = img.detach().cpu().numpy().transpose((1,2,0))
    temp = 255 * temp
    temp = temp.astype(np.uint8)
    temp = Image.fromarray(temp)
    return temp

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

class MemoryBankC:
    """Class that defines a memory bank of a single class"""
    def __init__(self, max_bank_size: int, c: int, root_dir, file_locks):
        """
        Args:
            c (int): chosen class
            mean( torch.Tensor): mean value for normalization
            std (torch.Tensor): std for normalization
            data_list (list): list of samples that contain the chosen class
        """
        self.memory_bank = SortedList()
        self.file_locks = file_locks
        self.max_bank_size = max_bank_size

        self.img_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.cheat_gt_dir = os.path.join(root_dir, "cheat_gt")
        self.prob_dir = os.path.join(root_dir, "prob")
        self.c = c

        self.bank_c_lock = Lock()

        for x in [self.img_dir, self.label_dir, self.cheat_gt_dir, self.prob_dir]:
            if not os.path.exists(x):
                os.makedirs(x)
            if not os.path.exists(osp.join(x, str(self.c))):
                os.mkdir(osp.join(x, str(self.c)))

    def find_index_of_file(self, file_name):
        """find the index of a filename in SortedList
        
        Args:
            s (str): file name
            sl (SortedList): sorted list of tuples (conf, filename)
        
        Returns:
            index of s in sl
        """
        file_names = [x[1].split("_local_iter_")[1] for x in self.memory_bank]
        try:
            idx = file_names.index(file_name)
        except ValueError as msg:
            return None
        return idx
    
    def check_add(self, conf):
        with self.bank_c_lock:
            if len(self.memory_bank) > 0 and len(self.memory_bank) >= self.max_bank_size:
                min_conf, _, _ = self.memory_bank[0]
            else:
                min_conf = -np.inf
            if conf > min_conf: # if conf is not greater than the min_conf of the memory bank, do nothing
                return True
            else:
                return False

    def add(self, conf, file_name):
        save_dir = ""
        add_bool = self.check_add(conf)
        # add_bool = True
        if add_bool: # if conf is not greater than the min_conf of the memory bank, do nothing

            with self.bank_c_lock: # lock entire bank_c here to ensure that there is no inconsistency between find idx and mb.pop(idx)
                file_name_stripped = file_name.split("_local_iter_")[1]
                idx = self.find_index_of_file(file_name_stripped)
                if idx is None:
                    # self.save_sample(img, label, prob_quantized, file_name, cheat_gt)
                    pass
                else:
                    # remove the old value if there is one
                    old_conf, old_file_name, old_save_dir = self.memory_bank.pop(idx)
            if idx is not None:
                if old_conf > conf:
                    conf = old_conf
                    file_name = old_file_name
                    save_dir = old_save_dir

                        

            with self.bank_c_lock:
                self.memory_bank.add((conf, file_name, save_dir))
                if len(self.memory_bank) > self.max_bank_size:
                    self.memory_bank.pop(0) # memory bank is sorted in ascending order, meaning least confident sample is first




    def __getitem__(self, idx):
        if idx < len(self.memory_bank) and len(self.memory_bank) > 0:
            return self.memory_bank[idx]
        else:
            return None
        
    def visualize_bank(self, palette, viz_dir, viz_class_name, it):
        for i, (conf, file_name, save_dir) in enumerate(self.memory_bank):
            file_name_c = osp.join(save_dir, file_name)

            cheat_gt = None
            label = Image.open(os.path.join(self.label_dir, save_dir, file_name))
            label = label.convert('P')
            label.putpalette(palette)
            img = Image.open(os.path.join(self.img_dir, save_dir, file_name))
            prob = Image.open(os.path.join(self.prob_dir, save_dir, file_name))
            if os.path.exists(os.path.join(self.cheat_gt_dir, save_dir, file_name)):
                cheat_gt = Image.open(os.path.join(self.cheat_gt_dir, save_dir, file_name))   

            temp1 = get_concat_v(img, label)
            if cheat_gt is not None:
                cheat_gt = Image.fromarray(np.array(cheat_gt).astype(np.uint8)).convert('P')
                cheat_gt.putpalette(palette)
                temp2 = get_concat_v(prob, cheat_gt)
            else:
                temp2 = get_concat_v(prob, prob)
            temp3 = get_concat_h(temp1, temp2)
            temp3 = temp3.resize((400, 400))

            if not os.path.exists(os.path.join(viz_dir, str(it), viz_class_name[int(self.c)])):
                os.makedirs(os.path.join(viz_dir, str(it), viz_class_name[int(self.c)]))
            temp3.save(os.path.join(viz_dir, str(it), viz_class_name[int(self.c)], str(conf) + file_name))

    def get_sample(self, idx):
        sample = self.__getitem__(idx)
        if sample is not None:
            cheat_gt = None
            conf, file_name, save_dir = sample
            file_name_c = osp.join(save_dir, file_name)
            with self.file_locks[file_name_c]: # lock the specific file_name
                label = Image.open(os.path.join(self.label_dir, save_dir, file_name))
                img = Image.open(os.path.join(self.img_dir, save_dir, file_name))
                prob = Image.open(os.path.join(self.prob_dir, save_dir, file_name))
                if os.path.exists(os.path.join(self.cheat_gt_dir, save_dir, file_name)):
                    cheat_gt = Image.open(os.path.join(self.cheat_gt_dir, save_dir, file_name))

            return img, label, prob, conf, cheat_gt
        else:
            return None

class MemoryBankAll:
    """Class that defines the memory bank of all classes."""
    def __init__(self, classes: List, mean, std, lock, tmp_dir, max_bank_size, crop_margins, rot_degree=20, min_scale=0.1):
        """
        Args:
            c (int): chosen class
            mean( torch.Tensor): mean value for normalization
            std (torch.Tensor): std for normalization
            data_list (list): list of samples that contain the chosen class
        """
        if not classes:
            self.classes = np.arange(19)
        else:
            self.classes = classes
        self.mean = mean
        self.std = std
        self.bank_lock = lock
        self.max_bank_size = max_bank_size
        self.rot_degree = rot_degree
        self.min_scale = min_scale

        viz_dir = tmp_dir[1]
        tmp_dir = tmp_dir[0]

        self.file_locks = {}

        self.crop_margins = crop_margins


        self.per_class_bank = dict((str(c), MemoryBankC(max_bank_size, c, tmp_dir, self.file_locks)) for c in self.classes)
        self.img_dir = os.path.join(tmp_dir, "images")
        self.label_dir = os.path.join(tmp_dir, "labels")
        self.cheat_gt_dir = os.path.join(tmp_dir, "cheat_gt")
        self.prob_dir = os.path.join(tmp_dir, "prob")
        self.viz_dir = os.path.join(viz_dir, "viz")
        if not os.path.exists(self.viz_dir):
            os.makedirs(self.viz_dir)
        
        self.viz_class_name = []
        for class_name in CityscapesDataset.CLASSES:
            class_name = class_name.replace(" ", "_")
            self.viz_class_name.append(class_name)
        

    def visualize_bank(self, it):
        palette = np.array(CityscapesDataset.PALETTE, dtype=np.uint8)
        for c, v in self.per_class_bank.items():
            with v.bank_c_lock:
                v.visualize_bank(palette, self.viz_dir, self.viz_class_name, it)



    def save_sample(self, img: Image.Image, label: Image.Image, prob_quantized: Image.Image, file_name: str, c_list: List, avg_conf_list: List, cheat_gt=None):
        """Save a new image and pseudo-label to disk, and add the avg confidence values per class to the memory bank.
        """
        
        if file_name in self.file_locks.keys():
            lock = self.file_locks[file_name]
        else:
            lock = Lock()
            self.file_locks[file_name] = lock

        with lock:
            img.save(os.path.join(self.img_dir, file_name))
            label.save(os.path.join(self.label_dir, file_name))
            prob_quantized.save(os.path.join(self.prob_dir, file_name))
            if cheat_gt is not None:
                cheat_gt.save(os.path.join(self.cheat_gt_dir, file_name))


        for c, avg_conf in zip(c_list, avg_conf_list):
            mb_c = self.per_class_bank[str(c)]
            mb_c.add(avg_conf, file_name)



    def get_sampling_prob(self, c, temperature):
        """Compute the sampling probability of each of the 'objects' in the class c memory bank.
        
        Args:
            c (int): class
            temperature (float): 'temperature' of the softmax
        
        Returns:
            Tuple(List, List)
        """
        memory_bank_c = self.per_class_bank[str(c)].memory_bank
        if len(memory_bank_c) > 0:
            # uniform sampling
            avg_conf = torch.tensor([memory_bank_c[x][0] for x in range(len(memory_bank_c))])
            if len(memory_bank_c) <= self.max_bank_size:
                sampling_prob = [1/len(memory_bank_c)] * len(memory_bank_c)
            else:
                sampling_prob = (len(memory_bank_c) - self.max_bank_size) * [0.] + self.max_bank_size * [1/self.max_bank_size] # The 50 most confident samples (last in memory bank) have prob 1/50
                # while the rest have prob 0.

            return sampling_prob, avg_conf.numpy().tolist()
        else:
            return None

    def __getitem__(self, c):
        """Get an object of class c.
        
        Args:
            c (int): class
            
        Returns:
            Tuple(Image.Image, Image.Image, np.ndarray)
        """
        img = None
        label = None
        cheat_gt = None
        sample = None
        prob = None
        with self.bank_lock: # lock the entire memory bank
            # draw samples from memory bank randomly by some probability distribution favoring confident samples.
            # TODO if I'm sampling from the entire memory bank, there is no need to keep it sorted.
            mb_c = self.per_class_bank[str(c)]
            with mb_c.bank_c_lock:
                res = self.get_sampling_prob(c, 0.01)
                if res is not None:
                    sampling_prob, _ = res
                    # idx = np.random.choice(np.arange(len(sampling_prob)), p=sampling_prob)
                    idx = torch.multinomial(torch.tensor(sampling_prob), 1).item()
                    # sample = self.per_class_bank[str(c)][idx]  
                    sample = mb_c.get_sample(idx)   
                    # mmcv.print_log(f'sample is None: {sample is None}', 'mmseg')
            
        # 
        if sample is not None:
            img, label, prob, conf, cheat_gt = sample

        if (img is not None) and (label is not None):
            extra_img, extra_label, r, cheat_gt, prob = self.random_scale_rot_pos(img, label, c, cheat_gt, prob)
            mask_extra = (np.array(extra_label) == c).astype(np.uint8)

            return extra_img, extra_label, mask_extra, cheat_gt, prob# , plot_img
        else:
            return None

    def __len__(self):
        return self.max_bank_size

    @staticmethod
    def norm(img, mean, std):
        return img.add(-mean).mul(1/std)
    @staticmethod
    def get_concat_h(im1, im2):
        dst = Image.new('RGB', (im1.width + im2.width, im1.height))
        dst.paste(im1, (0, 0))
        dst.paste(im2, (im1.width, 0))
        return dst

    @staticmethod
    def random_size(img, scale):
        new = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))
        return new

    def random_scale_rot_pos(self, img, label, c, cheat_gt=None, prob=None):
        temp = np.array(label)
        if self.crop_margins is not None:
            if self.crop_margins[0] > 0:
                temp[0:self.crop_margins[0], :] = 255
            if self.crop_margins[1] > 0:
                temp[-self.crop_margins[1]:, :] = 255
        label = Image.fromarray(temp)

        scale = self.min_scale + (1 - self.min_scale)*torch.rand(1).item()
        rot = torch.randint(low=-self.rot_degree, high=self.rot_degree + 1, size=(1,)).item()
        if (1.0 - scale)*img.width < 1.0 or (1.0 - scale)*img.height < 1.0:
            pos = (0,0)
        else:
            pos = (torch.randint(int((1.0 - scale)*img.width), size=(1,)).item(), torch.randint(int((1.0 - scale)*img.height), size=(1,)).item())
        flip = True if torch.rand(1).item() < 0.50 else False

        # create new image (placeholder)
        dst = Image.new('RGB', (img.width, img.height))
        # rescale image
        rescaled = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)), Image.BILINEAR)
        # flip image
        rescaled = rescaled if not flip else ImageOps.mirror(rescaled)
        # rotate image
        rescaled = rescaled.rotate(rot, Image.BILINEAR, expand=1)
        # paste onto placeholder (ensures that size is correct)
        dst.paste(rescaled, pos)

        # repeat the process for the label
        new_label = Image.new('L', (label.width, label.height), color=255)
        rescaled = label.resize((int(label.size[0] * scale), int(label.size[1] * scale)), Image.NEAREST)
        rescaled = rescaled if not flip else ImageOps.mirror(rescaled)
        rescaled = rescaled.rotate(rot, Image.NEAREST, expand=1, fillcolor=255)
        new_label.paste(rescaled, pos)

        # repeat for cheat_gt
        if cheat_gt is not None:
            new_cheat_gt = Image.new('L', (cheat_gt.width, cheat_gt.height), color=255)
            rescaled = cheat_gt.resize((int(cheat_gt.size[0] * scale), int(cheat_gt.size[1] * scale)), Image.NEAREST)
            rescaled = rescaled if not flip else ImageOps.mirror(rescaled)
            rescaled = rescaled.rotate(rot, Image.NEAREST, expand=1, fillcolor=255)
            new_cheat_gt.paste(rescaled, pos)
            cheat_gt = new_cheat_gt

        # repeat for prob
        if prob is not None:
            new_prob = Image.new('L', (prob.width, prob.height), color=0)
            rescaled = prob.resize((int(prob.size[0] * scale), int(prob.size[1] * scale)), Image.NEAREST)
            rescaled = rescaled if not flip else ImageOps.mirror(rescaled)
            rescaled = rescaled.rotate(rot, Image.NEAREST, expand=1, fillcolor=0)
            new_prob.paste(rescaled, pos)
            prob = new_prob

        return dst, new_label, rescaled, cheat_gt, prob
    

    def purge_unused_images(self):
        start = time.time_ns()
        n_selected = 0
        n_removed = 0
        remove_list = []
        for file_name, _ in self.file_locks.items():
            file_in_bank = False
            for c in self.classes:
                mb_c = self.per_class_bank[str(c)]
                file_name_stripped = file_name.split("_local_iter_")[1]
                idx = mb_c.find_index_of_file(file_name_stripped)
                if idx is not None:
                    file_in_bank = True
            if not file_in_bank:
                # remove file_name from self.file_locks and delete file from system
                remove_list.append(file_name)
                n_selected += 1
                for file_path in [os.path.join(x, file_name) for x in [self.img_dir, self.label_dir, self.prob_dir, self.cheat_gt_dir]]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        n_removed += 1
        for file_name in remove_list:
            del self.file_locks[file_name]

        end = time.time_ns()
        mmcv.print_log(f'Purging memory bank... n_selected: {n_selected}, n_removed: {n_removed}', 'mmseg')


@DATASETS.register_module()
class ECAP:
    """This class manages the memory bank with rare objects and creates augmented images from these."""
    def __init__(self, crop_margins, synthia, tmp_dir, mean, std, p_ea, start, rampup, ea_sigmoid, ea_beta, ea_gamma,
                 n_rounds=1, max_bank_size=50, rot_degree=20, min_scale=0.1,  classes=[], **kwargs):
        # TODO add an instance variable that can keep count of the number of threads. I.e. how many threads are creating new augmentations at the same time?
        assert kwargs.get('split') in [None, 'train']
        if 'split' in kwargs:
            kwargs.pop('split')

        assert tmp_dir is not None, "ECAP needs a directory to store the memory bank in"

        if not classes:
            classes = np.arange(19).tolist()
        self.classes = classes
        # self.classes = [17] # try with only one class for now

        self.n_rounds = n_rounds

        self.crop_margins = crop_margins
        if self.crop_margins is not None:
            if (self.crop_margins[0] ==0 and self.crop_margins[1] == 0):
                self.crop_margins = None # if both cropm argins are zero, we set it to None for easy handling downstream


        self.mean = torch.tensor(mean).reshape((1,3,1,1))
        self.std = torch.tensor(std).reshape((1,3,1,1))

        # lock for reading and writing to disk
        self.memory_bank = MemoryBankAll(classes=self.classes, mean=self.mean, std=self.std, lock=Lock(),
                                         tmp_dir=tmp_dir, max_bank_size=max_bank_size, rot_degree=rot_degree, min_scale=min_scale, crop_margins=crop_margins)

        self.sample_q = queue.Queue(maxsize=16)
        self.max_thread_count = 16

        # lock for accessing the queue in which augmented images and labels are stored
        self.lock = Lock()

        # self.thread_q = queue.Queue(maxsize=16) # specify max number of threads
        self.n_started = 0
        self.n_finished = 0

        self.temperature = 0.01

        self.expect_conf = 0

        self.local_iter = 0
        self.n_sampled = 0

        self.p_ea = p_ea
        self.start = start
        self.rampup = rampup
        self.ea_sigmoid = ea_sigmoid
        self.ea_beta = ea_beta
        self.ea_gamma = ea_gamma

        self.per_class_expectation = {c: 0.0 for c in np.arange(19).tolist()}

        self.synthia = synthia

        self.active_threads = []

    def visualize_bank(self, iter):
        self.memory_bank.visualize_bank(iter)

    def p_schedule(self, i):
        start = self.start
        rampup = self.rampup
        if i < start:
            p = 0.
        elif i < start + rampup:
            p = (i - start) / rampup
        else:
            p = 1
        p = self.p_ea * p
        
        if self.ea_sigmoid:
            # p = self.p_ea * torch.sigmoid((torch.tensor(self.expect_conf)-0.95)/0.025).numpy() # p increases as training progress
            p = self.p_ea * torch.sigmoid((torch.tensor(self.expect_conf) - self.ea_beta) / self.ea_gamma).numpy() # p increases as training progress

        return p

    def purge_bank(self):
        start = time.time_ns()

        purged = False
        while not purged:
            temp = []
            for i, t in enumerate(self.active_threads):
                if t.is_alive():
                    temp.append(self.active_threads[i])
            self.active_threads = temp
            if len(self.active_threads) == 0: # wait until all "add_sample_and_create_augmentation" threads have finished
                purged = True
                self.memory_bank.purge_unused_images()
        end = time.time_ns()
        mmcv.print_log(f'Purging memory bank took {(end - start)/1e9} seconds', 'mmseg')


    def manage_threads(self, target_img, pseudo_label, ema_softmax, file_name, cheat_gt=None):

        
        started_thread = False
        while started_thread == False: # try this (i.e. wait) until the thread can be started # TODO if the threads deadlock this would get stuck without raising any error
            temp = []
            for i, t in enumerate(self.active_threads):
                if t.is_alive():
                    temp.append(self.active_threads[i])
            self.active_threads = temp

            # if active_count() < self.max_thread_count: # only start a new thread if there are less than 16 active threads
            if len(self.active_threads) < self.max_thread_count: # only start a new thread if there are less than 16 active threads
                t = Thread(target=self.add_sample_and_create_augmentation, args=(target_img, pseudo_label, ema_softmax, file_name, cheat_gt))
                t.start()
                started_thread = True

    def create_augmentation(self):
        """Create an augmented image by sampling objects belonging to rare classes from the memory bank."""

        boost_classes, boost_prob = self.classes, [1/len(self.classes)] * len(self.classes) # uniform sampling

        img, label, mask, cheat_gt, prob = None, None, None, None, None
        # guaranteed_class = np.random.choice(np.arange(19), p=boost_prob) # TODO remove guaranteed_class and handle this special case in another way
        n_sampled_images = 0

        for round_n in range(self.n_rounds):
            # for c in np.random.permutation(self.classes): # shuffle so images doesnt get stacked in the same order every time
            for c in torch.randperm(len(self.classes)).numpy(): # shuffle so images doesnt get stacked in the same order every time
                rcs_class = boost_classes[c]
                rcs_prob = boost_prob[c] # * 5 # since I am allowed to sample from many different images I do not require the probability to sum to one

                p = self.p_schedule(self.local_iter)

                if self.synthia:
                    if self.per_class_expectation[rcs_class] <= 0.2: # for synthia, don't sample from memory bank with very low expected confidence score
                        continue
                if torch.rand(1).item() < rcs_prob * p: # with rcs probability * p, sample a random image containing the class and copy paste the class on top of current img
                    sample = self.memory_bank[rcs_class] # get an item from the memory bank of class rcs_class
                    if not sample is None:
                        n_sampled_images += 1
                        img_i, label_i, mask_i, cheat_gt_i, prob_i = sample
                        assert isinstance(img_i, Image.Image), f"Print img_i should be type Image but is type : {type(img_i)}"
                        assert isinstance(label_i, Image.Image), f"Print label_i should be type Image but is type : {type(label_i)}"
                        assert isinstance(mask_i, np.ndarray), f"Print img_i should be type np.ndarray but is type : {type(mask_i)}"

                        if img is None:
                            # img = Image.new('RGB', (img_i.width, img_i.height))
                            img = img_i
                            label = label_i
                            mask = mask_i
                            cheat_gt = cheat_gt_i
                            prob = prob_i
                            del img_i, label_i, mask_i, cheat_gt_i, prob_i
                        else:
                            img.paste(img_i, mask=Image.fromarray(255*(np.array(label_i) == rcs_class).astype(np.uint8)))
                            label.paste(label_i, mask=Image.fromarray(255*(np.array(label_i) == rcs_class).astype(np.uint8)))
                            mask = np.logical_or(mask, mask_i)
                            if cheat_gt_i is not None:
                                assert isinstance(cheat_gt_i, Image.Image), f"Print cheat_gt_i should be type Image but is type : {type(cheat_gt_i)}"
                                cheat_gt.paste(cheat_gt_i, mask=Image.fromarray(255*(np.array(label_i) == rcs_class).astype(np.uint8)))
                            if prob_i is not None:
                                assert isinstance(prob_i, Image.Image), f"Print prob_i should be type Image but is type : {type(prob_i)}"
                                prob.paste(prob_i, mask=Image.fromarray(255*(np.array(label_i) == rcs_class).astype(np.uint8)))

                            del img_i, label_i, mask_i, cheat_gt_i, prob_i


        # always add to sample_q, even if sample is None
        self.n_sampled += n_sampled_images
        lock_acquired = self.lock.acquire(timeout=10)
        try:
            if lock_acquired:
                if mask is not None:
                    mask = np.array(mask).astype(np.uint8)
                if prob is not None:
                    prob = np.array(prob) / 255 # convert back to probability values between 0 and 1
                if self.sample_q.full():
                    self.sample_q.get() # remove an item from q before adding a new one
                    self.sample_q.put((img, label, mask, cheat_gt, prob))
                else:
                    self.sample_q.put((img, label, mask, cheat_gt, prob))
                
                del img, label, mask, cheat_gt, prob
                
            else:
                mmcv.print_log(f'could not acquire lock in create_augmentation', 'mmseg')
                raise Exception("could not acquire lock in create_augmentation")
        finally:
            # always release the self.lock
            self.lock.release()

                    # else:
                    #     return None

    def add_sample(self, img, pseudo_label, pseudo_label_logits, file_name, cheat_gt):
        assert img.dim() == 3, f"img dimension must equal 3, but has shape {img.shape}"
        assert pseudo_label.dim() == 2, f"pseudo_label dimension must equal 2, but has shape {pseudo_label.shape}"
        assert pseudo_label_logits.dim() == 3, f"pseudo_label_logits dimension must equal 3, but has shape {pseudo_label_logits.shape}"

        teacher_prob, _ = torch.max(pseudo_label_logits, dim=0)
        teacher_prob = teacher_prob.detach().cpu().numpy() * 255
        teacher_prob = teacher_prob.astype(np.uint8)
        teacher_prob_img = Image.fromarray(teacher_prob)
        pseudo_label = pseudo_label.detach().cpu().numpy()

        cheat_gt_img = None
        if cheat_gt is not None:
            cheat_gt = cheat_gt.detach().cpu().numpy().astype(np.uint8)
            cheat_gt_img = Image.fromarray(cheat_gt)


        denorm_img = tensor_to_image(img, self.mean, self.std)
        del img # delete img to release gpu memory
        pseudo_label_img = Image.fromarray(pseudo_label.astype(np.uint8))
        pseudo_label_logits = pseudo_label_logits.detach().cpu().numpy()

        # don't take the top or bottom part of the image into account. Here we know that the labels are not trustworthy.
        # arr_crop = pseudo_label_logits[:, 30:784, :]
        if self.crop_margins is not None:
            if self.crop_margins[0] > 0 and self.crop_margins[1] > 0:
                arr_crop = pseudo_label_logits[:, self.crop_margins[0]:-self.crop_margins[1], :]
            elif self.crop_margins[0] > 0 and self.crop_margins[1] == 0:
                arr_crop = pseudo_label_logits[:, self.crop_margins[0]:, :]
            elif self.crop_margins[0] == 0 and self.crop_margins[1] > 0:
                arr_crop = pseudo_label_logits[:, self.crop_margins[0]:-self.crop_margins[1], :]
        else:
            arr_crop = pseudo_label_logits
        avg_conf_list = []
        c_list = []
        c_counts = []
        for c in self.classes:
            # pseudo_labels_c = (pseudo_label[30:784, :] == c).astype(np.uint8)
            if self.crop_margins is not None:
                if self.crop_margins[0] > 0 and self.crop_margins[1] > 0:
                    pseudo_labels_c = (pseudo_label[self.crop_margins[0]:-self.crop_margins[1], :] == c).astype(np.uint8)
                elif self.crop_margins[0] > 0 and self.crop_margins[1] == 0:
                    pseudo_labels_c = (pseudo_label[self.crop_margins[0]:, :] == c).astype(np.uint8)
                elif self.crop_margins[0] == 0 and self.crop_margins[1] > 0:
                    pseudo_labels_c = (pseudo_label[self.crop_margins[0]:-self.crop_margins[1], :] == c).astype(np.uint8)
            else:
                pseudo_labels_c = (pseudo_label == c).astype(np.uint8)
            n_predicted_pixels_class_c = np.sum(pseudo_labels_c) # number of pixels that belong to class c according to pseudo-label
            if n_predicted_pixels_class_c > 0:
                c_counts.append(n_predicted_pixels_class_c)
                predicted_prob_at_c = arr_crop[c,:,:] * pseudo_labels_c
                avg_conf = np.sum(predicted_prob_at_c) / n_predicted_pixels_class_c
                avg_conf_list.append(avg_conf)
                c_list.append(c)
                del predicted_prob_at_c
            del pseudo_labels_c


        del pseudo_label, pseudo_label_logits, arr_crop
        # the bank_lock is aquired and released insie save_sample
        self.memory_bank.save_sample(denorm_img, pseudo_label_img, teacher_prob_img, file_name, c_list, avg_conf_list, cheat_gt_img)
        del denorm_img, pseudo_label_img, c_list, avg_conf_list


    def add_sample_and_create_augmentation(self, img, pseudo_label, pseudo_label_logits, file_name, cheat_gt):
        """Save new pseudo-label and image to disk and add the occuring classes to the memory banks along with the avg confidence."""
        # add new 
        self.n_started += 1
        
        # always add the new sample to the memory bank
        self.add_sample(img, pseudo_label, pseudo_label_logits, file_name, cheat_gt)
        self.create_augmentation()

        self.n_finished += 1


    def __getitem__(self, idx):
        """
        Args:
        
        Returns:
            (img: Image.Image, label: Image.Image, mask: np.ndarray, cheat_gt: Image.Image, prob: np.ndarray)
        """
        with self.lock:
            if not self.sample_q.empty():
                return self.sample_q.get()
            else:
                return None


    def compute_confidence_expectation(self):
        """Compute the expected confidence of a sample drawn from the memory bank according to class and sample sampling probabilities."""
        boost_classes, boost_prob = self.classes, [1/len(self.classes)] * len(self.classes) # uniform sampling

        expectation = 0
        per_class_expectation = {}
        with self.memory_bank.bank_lock:
            for c, p_c in zip(boost_classes, boost_prob):
                res = self.memory_bank.get_sampling_prob(c, 0.01)
                if res is not None:
                    p_co, conf = res
                    expectation_c = np.sum(np.multiply(p_co, conf))
                    per_class_expectation[c] = expectation_c
                    expectation += p_c * expectation_c
                else:
                    per_class_expectation[c] = 0.0
        self.expect_conf = expectation
        self.per_class_expectation = per_class_expectation

        return expectation, per_class_expectation

    @staticmethod
    def norm(img, mean, std):
        return img.add(-mean).mul(1/std)



