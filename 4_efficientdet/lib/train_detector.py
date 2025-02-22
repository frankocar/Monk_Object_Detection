import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from src.dataset import CocoDataset, Resizer, Normalizer, Augmenter, collater
from src.model import EfficientDet
from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.autonotebook import tqdm


class Detector():
    '''
    Class to train a detector

    Args:
        verbose (int): Set verbosity levels
                        0 - Print Nothing
                        1 - Print desired details
    '''
    def __init__(self, verbose=1):
        self.system_dict = {};
        self.system_dict["verbose"] = verbose;
        self.system_dict["local"] = {};
        self.system_dict["dataset"] = {};
        self.system_dict["dataset"]["train"] = {};
        self.system_dict["dataset"]["val"] = {};
        self.system_dict["dataset"]["val"]["status"] = False;

        self.system_dict["params"] = {};
        self.system_dict["params"]["image_size"] = 1024;
        self.system_dict["params"]["batch_size"] = 8;
        self.system_dict["params"]["num_workers"] = 3;
        self.system_dict["params"]["use_gpu"] = True;
        self.system_dict["params"]["gpu_devices"] = [0];
        self.system_dict["params"]["lr"] = 0.0001;
        self.system_dict["params"]["num_epochs"] = 10;
        self.system_dict["params"]["val_interval"] = 1;
        self.system_dict["params"]["es_min_delta"] = 0.0;
        self.system_dict["params"]["es_patience"] = 0;


        self.system_dict["output"] = {};
        self.system_dict["output"]["log_path"] = "tensorboard/signatrix_efficientdet_coco";
        self.system_dict["output"]["saved_path"] = "trained/";
        self.system_dict["output"]["best_epoch"] = 0;
        self.system_dict["output"]["best_loss"] = 1e5;



    def Train_Dataset(self, root_dir, coco_dir, img_dir, set_dir, batch_size=8, image_size=1024, use_gpu=True, num_workers=3):
        '''
        User function: Set training dataset parameters

        Dataset Directory Structure

                   root_dir
                      |
                      |------coco_dir 
                      |         |
                      |         |----img_dir
                      |                |
                      |                |------<set_dir_train> (set_dir) (Train)
                      |                         |
                      |                         |---------img1.jpg
                      |                         |---------img2.jpg
                      |                         |---------..........(and so on)  
                      |
                      |
                      |         |---annotations 
                      |         |----|
                      |              |--------------------instances_Train.json  (instances_<set_dir_train>.json)
                      |              |--------------------classes.txt
                      
                      
             - instances_Train.json -> In proper COCO format
             - classes.txt          -> A list of classes in alphabetical order
             

            For TrainSet
             - root_dir = "../sample_dataset";
             - coco_dir = "kangaroo";
             - img_dir = "images";
             - set_dir = "Train";
            
             
            Note: Annotation file name too coincides against the set_dir

        Args:
            root_dir (str): Path to root directory containing coco_dir
            coco_dir (str): Name of coco_dir containing image folder and annotation folder
            img_dir (str): Name of folder containing all training and validation folders
            set_dir (str): Name of folder containing all training images
            batch_size (int): Mini batch sampling size for training epochs
            image_size (int): Either of [512, 300]
            use_gpu (bool): If True use GPU else run on CPU
            num_workers (int): Number of parallel processors for data loader 

        Returns:
            None
        '''
        self.system_dict["dataset"]["train"]["root_dir"] = root_dir;
        self.system_dict["dataset"]["train"]["coco_dir"] = coco_dir;
        self.system_dict["dataset"]["train"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["train"]["set_dir"] = set_dir;


        self.system_dict["params"]["batch_size"] = batch_size;
        self.system_dict["params"]["image_size"] = image_size;
        self.system_dict["params"]["use_gpu"] = use_gpu;
        self.system_dict["params"]["num_workers"] = num_workers;

        if(self.system_dict["params"]["use_gpu"]):
            if torch.cuda.is_available():
                self.system_dict["local"]["num_gpus"] = torch.cuda.device_count()
                torch.cuda.manual_seed(123)
            else:
                torch.manual_seed(123)

        self.system_dict["local"]["training_params"] = {"batch_size": self.system_dict["params"]["batch_size"] * self.system_dict["local"]["num_gpus"],
                                                           "shuffle": True,
                                                           "drop_last": True,
                                                           "collate_fn": collater,
                                                           "num_workers": self.system_dict["params"]["num_workers"]}

        self.system_dict["local"]["training_set"] = CocoDataset(root_dir=self.system_dict["dataset"]["train"]["root_dir"] + "/" + self.system_dict["dataset"]["train"]["coco_dir"],
                                                            img_dir = self.system_dict["dataset"]["train"]["img_dir"],
                                                            set_dir = self.system_dict["dataset"]["train"]["set_dir"],
                                                            transform = transforms.Compose([Normalizer(), Augmenter(), Resizer(common_size = self.system_dict["params"]["image_size"])])
                                                            )
        
        self.system_dict["local"]["training_generator"] = DataLoader(self.system_dict["local"]["training_set"], 
                                                                    **self.system_dict["local"]["training_params"]);


    def Val_Dataset(self, root_dir, coco_dir, img_dir, set_dir):
        '''
        User function: Set training dataset parameters

        Dataset Directory Structure

                   root_dir
                      |
                      |------coco_dir 
                      |         |
                      |         |----img_dir
                      |                |
                      |                |------<set_dir_val> (set_dir) (Validation)
                      |                         |
                      |                         |---------img1.jpg
                      |                         |---------img2.jpg
                      |                         |---------..........(and so on)  
                      |
                      |
                      |         |---annotations 
                      |         |----|
                      |              |--------------------instances_Val.json  (instances_<set_dir_val>.json)
                      |              |--------------------classes.txt
                      
                      
             - instances_Train.json -> In proper COCO format
             - classes.txt          -> A list of classes in alphabetical order

             
            For ValSet
             - root_dir = "..sample_dataset";
             - coco_dir = "kangaroo";
             - img_dir = "images";
             - set_dir = "Val";
             
             Note: Annotation file name too coincides against the set_dir

        Args:
            root_dir (str): Path to root directory containing coco_dir
            coco_dir (str): Name of coco_dir containing image folder and annotation folder
            img_dir (str): Name of folder containing all training and validation folders
            set_dir (str): Name of folder containing all validation images

        Returns:
            None
        '''
        self.system_dict["dataset"]["val"]["status"] = True;
        self.system_dict["dataset"]["val"]["root_dir"] = root_dir;
        self.system_dict["dataset"]["val"]["coco_dir"] = coco_dir;
        self.system_dict["dataset"]["val"]["img_dir"] = img_dir;
        self.system_dict["dataset"]["val"]["set_dir"] = set_dir;     

        self.system_dict["local"]["val_params"] = {"batch_size": self.system_dict["params"]["batch_size"],
                                                   "shuffle": False,
                                                   "drop_last": False,
                                                   "collate_fn": collater,
                                                   "num_workers": self.system_dict["params"]["num_workers"]}

        self.system_dict["local"]["val_set"] = CocoDataset(root_dir=self.system_dict["dataset"]["val"]["root_dir"] + "/" + self.system_dict["dataset"]["val"]["coco_dir"], 
                                                    img_dir = self.system_dict["dataset"]["val"]["img_dir"],
                                                    set_dir = self.system_dict["dataset"]["val"]["set_dir"],
                                                    transform=transforms.Compose([Normalizer(), Resizer(common_size = self.system_dict["params"]["image_size"])]))
        
        self.system_dict["local"]["test_generator"] = DataLoader(self.system_dict["local"]["val_set"], 
                                                                **self.system_dict["local"]["val_params"])

    
    #efficientnet-b0;
    #efficientnet-b1;
    #efficientnet-b2;
    #efficientnet-b3;
    #efficientnet-b4;
    #efficientnet-b5;
    #efficientnet-b6;
    #efficientnet-b7;
    #efficientnet-b8;
    def Model(self, model_name="efficientnet-b0", gpu_devices=[0], load_pretrained_model_from=None):
        '''
        User function: Set Model parameters

        Args:
            gpu_devices (list): List of GPU Device IDs to be used in training

        Returns:
            None
        '''
        if(not load_pretrained_model_from):
            num_classes = self.system_dict["local"]["training_set"].num_classes();
            coeff = int(model_name[-1])
            efficientdet = EfficientDet(num_classes=num_classes, compound_coef=coeff, model_name=model_name);

            if self.system_dict["params"]["use_gpu"]:
                self.system_dict["params"]["gpu_devices"] = gpu_devices
                if len(self.system_dict["params"]["gpu_devices"])==1:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.system_dict["params"]["gpu_devices"][0])
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in self.system_dict["params"]["gpu_devices"]])
                self.system_dict["local"]["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
                efficientdet = efficientdet.to(self.system_dict["local"]["device"])
                efficientdet= torch.nn.DataParallel(efficientdet).to(self.system_dict["local"]["device"])

            self.system_dict["local"]["model"] = efficientdet;
            self.system_dict["local"]["model"].train();
        else:
            efficientdet = torch.load(load_pretrained_model_from).module
            if self.system_dict["params"]["use_gpu"]:
                self.system_dict["params"]["gpu_devices"] = gpu_devices
                if len(self.system_dict["params"]["gpu_devices"])==1:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.system_dict["params"]["gpu_devices"][0])
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in self.system_dict["params"]["gpu_devices"]])
                self.system_dict["local"]["device"] = 'cuda' if torch.cuda.is_available() else 'cpu'
                efficientdet = efficientdet.to(self.system_dict["local"]["device"])
                efficientdet= torch.nn.DataParallel(efficientdet).to(self.system_dict["local"]["device"])
            
            self.system_dict["local"]["model"] = efficientdet;
            self.system_dict["local"]["model"].train();
       


    def Set_Hyperparams(self, lr=0.0001, val_interval=1, es_min_delta=0.0, es_patience=0):
        '''
        User function: Set hyper parameters

        Args:
            lr (float): Initial learning rate for training
            val_interval (int): Post specified number of training epochs, a validation epoch will be carried out
            es_min_delta (float): Loss detla value, if loss doesnn't change more than this value for "es_patience" number of epochs, training will be stopped early
            es_patience (int): If loss doesnn't change more than this "es_min_delta" value for "es_patience" number of epochs, training will be stopped early

        Returns:
            None
        '''
        self.system_dict["params"]["lr"] = lr;
        self.system_dict["params"]["val_interval"] = val_interval;
        self.system_dict["params"]["es_min_delta"] = es_min_delta;
        self.system_dict["params"]["es_patience"] = es_patience;


        self.system_dict["local"]["optimizer"] = torch.optim.Adam(self.system_dict["local"]["model"].parameters(), 
                                                                    self.system_dict["params"]["lr"]);

        self.system_dict["local"]["scheduler"] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.system_dict["local"]["optimizer"], 
                                                                    patience=3, verbose=True)


    def Train(self, num_epochs=2, model_output_dir="trained/"):
        '''
        User function: Start training

        Args:
            num_epochs (int): Number of epochs to train for
            model_output_dir (str): Path to directory where all trained models will be saved

        Returns:
            None
        '''
        self.system_dict["output"]["log_path"] = "tensorboard/signatrix_efficientdet_coco";
        self.system_dict["output"]["saved_path"] = model_output_dir;
        self.system_dict["params"]["num_epochs"] = num_epochs;

        if os.path.isdir(self.system_dict["output"]["log_path"]):
            shutil.rmtree(self.system_dict["output"]["log_path"])
        os.makedirs(self.system_dict["output"]["log_path"])

        if os.path.isdir(self.system_dict["output"]["saved_path"]):
            shutil.rmtree(self.system_dict["output"]["saved_path"])
        os.makedirs(self.system_dict["output"]["saved_path"])

        writer = SummaryWriter(self.system_dict["output"]["log_path"])

        num_iter_per_epoch = len(self.system_dict["local"]["training_generator"])

        if(self.system_dict["dataset"]["val"]["status"]):
            
            for epoch in range(self.system_dict["params"]["num_epochs"]):
                self.system_dict["local"]["model"].train()

                epoch_loss = []
                progress_bar = tqdm(self.system_dict["local"]["training_generator"])
                for iter, data in enumerate(progress_bar):
                    try:
                        self.system_dict["local"]["optimizer"].zero_grad()
                        if torch.cuda.is_available():
                            cls_loss, reg_loss = self.system_dict["local"]["model"]([data['img'].to(self.system_dict["local"]["device"]).float(), data['annot'].to(self.system_dict["local"]["device"])])
                        else:
                            cls_loss, reg_loss = self.system_dict["local"]["model"]([data['img'].float(), data['annot']])

                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()
                        loss = cls_loss + reg_loss
                        if loss == 0:
                            continue
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.system_dict["local"]["model"].parameters(), 0.1)
                        self.system_dict["local"]["optimizer"].step()
                        epoch_loss.append(float(loss))
                        total_loss = np.mean(epoch_loss)

                        progress_bar.set_description(
                            'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Batch loss: {:.5f} Total loss: {:.5f}'.format(
                                epoch + 1, self.system_dict["params"]["num_epochs"], iter + 1, num_iter_per_epoch, cls_loss, reg_loss, loss,
                                total_loss))
                        writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
                        writer.add_scalar('Train/Regression_loss', reg_loss, epoch * num_iter_per_epoch + iter)
                        writer.add_scalar('Train/Classfication_loss (focal loss)', cls_loss, epoch * num_iter_per_epoch + iter)

                    except Exception as e:
                        print(e)
                        continue
                self.system_dict["local"]["scheduler"].step(np.mean(epoch_loss))

                if epoch % self.system_dict["params"]["val_interval"] == 0:

                    self.system_dict["local"]["model"].eval()
                    loss_regression_ls = []
                    loss_classification_ls = []
                    for iter, data in enumerate(self.system_dict["local"]["test_generator"]):
                        with torch.no_grad():
                            if torch.cuda.is_available():
                                cls_loss, reg_loss = self.system_dict["local"]["model"]([data['img'].to(self.system_dict["local"]["device"]).float(), data['annot'].to(self.system_dict["local"]["device"])])
                            else:
                                cls_loss, reg_loss = self.system_dict["local"]["model"]([data['img'].float(), data['annot']])

                            cls_loss = cls_loss.mean()
                            reg_loss = reg_loss.mean()

                            loss_classification_ls.append(float(cls_loss))
                            loss_regression_ls.append(float(reg_loss))

                    cls_loss = np.mean(loss_classification_ls)
                    reg_loss = np.mean(loss_regression_ls)
                    loss = cls_loss + reg_loss

                    print(
                        'Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                            epoch + 1, self.system_dict["params"]["num_epochs"], cls_loss, reg_loss,
                            np.mean(loss)))
                    writer.add_scalar('Val/Total_loss', loss, epoch)
                    writer.add_scalar('Val/Regression_loss', reg_loss, epoch)
                    writer.add_scalar('Val/Classfication_loss (focal loss)', cls_loss, epoch)

                    if loss + self.system_dict["params"]["es_min_delta"] < self.system_dict["output"]["best_loss"]:
                        self.system_dict["output"]["best_loss"] = loss
                        self.system_dict["output"]["best_epoch"] = epoch
                        torch.save(self.system_dict["local"]["model"], 
                            os.path.join(self.system_dict["output"]["saved_path"], "signatrix_efficientdet_coco.pth"))

                        dummy_input = torch.rand(1, 3, 1024, 1024)
                        if torch.cuda.is_available():
                            dummy_input = dummy_input.cuda()
                        if isinstance(self.system_dict["local"]["model"], nn.DataParallel):
                            self.system_dict["local"]["model"].module.backbone_net.model.set_swish(memory_efficient=False)
                            self.system_dict["local"]["model"].module.backbone_net.model.set_swish(memory_efficient=True)
                        else:
                            self.system_dict["local"]["model"].backbone_net.model.set_swish(memory_efficient=False)
                            self.system_dict["local"]["model"].backbone_net.model.set_swish(memory_efficient=True)

                    # Early stopping
                    if epoch - self.system_dict["output"]["best_epoch"] > self.system_dict["params"]["es_patience"] > 0:
                        print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
                        break

        else:
            for epoch in range(self.system_dict["params"]["num_epochs"]):
                self.system_dict["local"]["model"].train()

                epoch_loss = []
                progress_bar = tqdm(self.system_dict["local"]["training_generator"])
                for iter, data in enumerate(progress_bar):
                    try:
                        self.system_dict["local"]["optimizer"].zero_grad()
                        if torch.cuda.is_available():
                            cls_loss, reg_loss = self.system_dict["local"]["model"]([data['img'].to(self.system_dict["local"]["device"]).float(), data['annot'].to(self.system_dict["local"]["device"])])
                        else:
                            cls_loss, reg_loss = self.system_dict["local"]["model"]([data['img'].float(), data['annot']])

                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()
                        loss = cls_loss + reg_loss
                        if loss == 0:
                            continue
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.system_dict["local"]["model"].parameters(), 0.1)
                        self.system_dict["local"]["optimizer"].step()
                        epoch_loss.append(float(loss))
                        total_loss = np.mean(epoch_loss)

                        progress_bar.set_description(
                            'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Batch loss: {:.5f} Total loss: {:.5f}'.format(
                                epoch + 1, self.system_dict["params"]["num_epochs"], iter + 1, num_iter_per_epoch, cls_loss, reg_loss, loss,
                                total_loss))
                        writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
                        writer.add_scalar('Train/Regression_loss', reg_loss, epoch * num_iter_per_epoch + iter)
                        writer.add_scalar('Train/Classfication_loss (focal loss)', cls_loss, epoch * num_iter_per_epoch + iter)

                    except Exception as e:
                        print(e)
                        continue
                self.system_dict["local"]["scheduler"].step(np.mean(epoch_loss))


                torch.save(self.system_dict["local"]["model"], 
                    os.path.join(self.system_dict["output"]["saved_path"], "signatrix_efficientdet_coco.pth"))

                dummy_input = torch.rand(1, 3, 1024, 1024)
                if torch.cuda.is_available():
                    dummy_input = dummy_input.to(self.system_dict["local"]["device"])
                if isinstance(self.system_dict["local"]["model"], nn.DataParallel):
                    self.system_dict["local"]["model"].module.backbone_net.model.set_swish(memory_efficient=False)
                    self.system_dict["local"]["model"].module.backbone_net.model.set_swish(memory_efficient=True)
                else:
                    self.system_dict["local"]["model"].backbone_net.model.set_swish(memory_efficient=False)
                    self.system_dict["local"]["model"].backbone_net.model.set_swish(memory_efficient=True)


        writer.close()
