#!/usr/bin/env python3

import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision.io as io
import os
import json
from torchvision.io.video import re
from tqdm import tqdm
import csv
import timm
import wandb
import matplotlib.pyplot as plt
import numpy as np


def f1_score(true_labels, pred_labels):
    """
    Calculate the custom F1-score.

    Parameters:
    true_labels (list or array-like): The ground truth labels.
    pred_labels (list or array-like): The predicted labels.

    Returns:
    float: The custom F1-score.
    """
    # Convert input lists to NumPy arrays
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # Calculate True Positives (TP)
    TP = np.sum((true_labels == 1) & (pred_labels == 1))
    
    # Calculate False Positives (FP)
    FP = np.sum((true_labels == 0) & (pred_labels == 1))
    
    # Calculate False Negatives (FN)
    FN = np.sum((true_labels == 1) & (pred_labels == 0))
    
    # Calculate custom F1-score
    denominator = 2 * TP + FP + FN
    if denominator == 0:
        return 0.0  # Avoid division by zero
    else:
        f1_score = (2 * TP) / denominator
        return f1_score




from PIL import Image
import torchvision.transforms.v2 as transforms

# UTILITIES

def extract_frames(video_path, nb_frames=10, delta=1, timeit=False):
    # use time to measure the time it takes to resize a video
    t1 = time.time()
    reader = io.VideoReader(video_path)
    # take 10 frames uniformly sampled from the video
    frames = []
    for i in range(nb_frames):
        reader.seek(delta)
        frame = next(reader)
        frames.append(frame['data'])
    t2 = time.time()     
    video = torch.stack(frames)
    if timeit:
        print(f"read: {t2-t1}")
    return video

def smart_resize(data, size): # kudos louis
    # Prends un tensor de shape [...,C,H,W] et le resize en [...C,size,size]
    # x, y, height et width servent a faire un crop avant de resize

    full_height = data.shape[-2]
    full_width = data.shape[-1]

    if full_height > full_width:
        alt_height = size
        alt_width = int(full_width / (full_height / size))
    elif full_height < full_width:
        alt_height = int(full_height / (full_width / size))
        alt_width = size
    else:
        alt_height = size
        alt_width = size
    tr = transforms.Compose([
        transforms.Resize((alt_height, alt_width)),
        transforms.CenterCrop(size)
    ])
    return tr(data)

def resize_data(data, new_height, new_width, x=0, y=0, height=None, width=None):
    # Prends un tensor de shape [...,C,H,W] et le resize en [C,new_height,new_width]
    # x, y, height et width servent a faire un crop avant de resize

    full_height = data.shape[-2]
    full_width = data.shape[-1]
    height = full_height - y if height is None else height
    width = full_width -x if width is None else width

    ratio = new_height/new_width
    if height/width > ratio:
        expand_height = height
        expand_width = int(height / ratio)
    elif height/width < ratio:
        expand_height = int(width * ratio)
        expand_width = width
    else:
        expand_height = height
        expand_width = width
    tr = transforms.Compose([
        transforms.CenterCrop((expand_height, expand_width)),
        transforms.Resize((new_height, new_width))
    ])
    x = data[...,y:min(y+height, full_height), x:min(x+width, full_width)].clone()
    return tr(x)


# SETUP DATASET

dataset_dir = "/raid/datasets/hackathon2024"
root_dir = os.path.expanduser("~/automathon-2024")
nb_frames = 10

## MAKE RESIZED DATASET
resized_dir = os.path.join(dataset_dir, "resized_dataset")
"""
create_small_dataset = False
errors = []
if not os.path.exists(resized_dir) or create_small_dataset:
    os.mkdir(resized_dir)
    os.mkdir(os.path.join(resized_dir, "train_dataset"))
    os.mkdir(os.path.join(resized_dir, "test_dataset"))
    os.mkdir(os.path.join(resized_dir, "experimental_dataset"))
    train_files = [f for f in os.listdir(os.path.join(dataset_dir, "train_dataset")) if f.endswith('.mp4')]
    test_files = [f for f in os.listdir(os.path.join(dataset_dir, "test_dataset")) if f.endswith('.mp4')]
    experimental_files = [f for f in os.listdir(os.path.join(dataset_dir, "experimental_dataset")) if f.endswith('.mp4')]
    def resize(in_video_path, out_video_path, nb_frames=10):
        video = extract_frames(in_video_path, nb_frames=nb_frames)
        t1 = time.time()
        #video, audio, info = io.read_video(in_video_path, pts_unit='sec', start_pts=0, end_pts=10, output_format='TCHW')
        video = smart_resize(video, 256)
        t2 = time.time()
        torch.save(video, out_video_path)
        t3 = time.time()
        print(f"resize: {t2-t1}\nsave: {t3-t2}")
        #video = video.permute(0,2,3,1)
        #io.write_video(video_path, video, 15, video_codec='h264')

    
    for f in tqdm(train_files):
        in_video_path = os.path.join(dataset_dir, "train_dataset", f)
        out_video_path = os.path.join(resized_dir, "train_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from train")
    
    for f in tqdm(test_files):
        in_video_path = os.path.join(dataset_dir, "test_dataset", f)
        out_video_path = os.path.join(resized_dir, "test_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from test")
    for f in tqdm(experimental_files):
        in_video_path = os.path.join(dataset_dir, "experimental_dataset", f)
        out_video_path = os.path.join(resized_dir, "experimental_dataset", f[:-3] + "pt")
        try:
            resize(in_video_path, out_video_path)
        except Exception as e:
            errors.append((f, e))
        print(f"resized {f} from experimental")
    os.system(f"cp {os.path.join(dataset_dir, 'train_dataset', 'metadata.json')} {os.path.join(resized_dir, 'train_dataset', 'metadata.json')}")
    os.system(f"cp {os.path.join(dataset_dir, 'dataset.csv')} {os.path.join(resized_dir, 'dataset.csv')}")
    os.system(f"cp {os.path.join(dataset_dir, 'experimental_dataset', 'metadata.json')} {os.path.join(resized_dir, 'experimental_dataset', 'metadata.json')}")
    if errors:
        print(errors)
"""
use_small_dataset = True
if use_small_dataset:
    dataset_dir = resized_dir

nb_frames = 10

class VideoDataset(Dataset):
    """
    This Dataset takes a video and returns a tensor of shape [10, 3, 256, 256]
    That is 10 colored frames of 256x256 pixels.
    """
    def __init__(self, root_dir, dataset_choice="train", nb_frames=10):
        super().__init__()
        self.dataset_choice = dataset_choice
        if  self.dataset_choice == "train":
            self.root_dir = os.path.join(root_dir, "train_dataset")
        elif  self.dataset_choice == "test":
            self.root_dir = os.path.join(root_dir, "test_dataset")
        elif  self.dataset_choice == "experimental":
            self.root_dir = os.path.join(root_dir, "experimental_dataset")
        else:
            raise ValueError("choice must be 'train', 'test' or 'experimental'")

        with open(os.path.join(root_dir, "dataset.csv"), 'r') as file:
            reader = csv.reader(file)
            # read dataset.csv with id,label columns to create
            # a dict which associated label: id
            self.ids = {row[1][:-3] + "pt" : row[0] for row in reader}

        if self.dataset_choice == "test":
            self.data = None
        else:
            with open(os.path.join(self.root_dir, "metadata.json"), 'r') as file:
                self.data= json.load(file)
                self.data = {k[:-3] + "pt" : (torch.tensor(float(1)) if v == 'FAKE' else torch.tensor(float(0))) for k, v in self.data.items()}

        #self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.mp4')]
        self.video_files = [f for f in os.listdir(self.root_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = os.path.join(self.root_dir, self.video_files[idx])
        #video, audio, info = io.read_video(video_path, pts_unit='sec')
        #video = extract_frames(video_path)
        video = torch.load(video_path)

        """
        video = video.permute(0,3,1,2)
        length = video.shape[0]
        video = video[[i*(length//(nb_frames)) for i in range(nb_frames)]]
        """
        # resize the data into a reglar shape of 256x256 and normalize it
        #video = smart_resize(video, 256) / 255
        video = video / 255

        ID = self.ids[self.video_files[idx]]
        if self.dataset_choice == "test":
            return video, ID
        else:
            label = self.data[self.video_files[idx]]
            return video, label, ID


dataset_dir='dataset_tensor'
# train_dataset = VideoDataset(dataset_dir, dataset_choice="train", nb_frames=nb_frames)
test_dataset = VideoDataset(dataset_dir, dataset_choice="test", nb_frames=nb_frames)
experimental_dataset = VideoDataset(dataset_dir, dataset_choice="experimental", nb_frames=nb_frames)
train_dataset = experimental_dataset
print(f"Length of experimental_dataset: {len(experimental_dataset)}")

# MODELE

class DeepfakeDetector(nn.Module):
    def __init__(self, nb_frames=10):
        super().__init__()
        self.dense = nn.Linear(nb_frames*3*256*256,1)
        self.flat = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.flat(x)
        y = self.dense(y)
        y = self.sigmoid(y)
        return y

# LOGGING

# wandb.login(key="a446d513570a79c857317c3000584c5f6d6224f0")

# run = wandb.init(
#     project="automathon"
# )

# ENTRAINEMENT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
loss_fn = nn.MSELoss()
model = DeepfakeDetector().to(device)
print("Training model:")
summary(model, input_size=(batch_size, 3, 10, 256, 256))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 5
print(f"Length of train_dataset: {len(train_dataset)}")
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#loader = DataLoader(experimental_dataset, batch_size=2, shuffle=True)

train_f1_scores = []

print("Training...")
for epoch in range(epochs):
    true_labels = []
    pred_labels = []
    
    for sample in tqdm(loader):
        optimizer.zero_grad()
        X, label, ID = sample
        X = X.to(device)
        label = label.to(device)
        label_pred = model(X)
        label = torch.unsqueeze(label, dim=1)
        loss = loss_fn(label, label_pred)
        loss.backward()
        optimizer.step()
        # run.log({"loss": loss.item(), "epoch": epoch})
        print({"loss": loss.item(), "epoch": epoch})
        
        # Collect true and predicted labels
        pred = (label_pred > 0.5).long().cpu().detach().numpy()
        true_labels.extend(label.cpu().detach().numpy())
        pred_labels.extend(pred)
    
    print('true_labels = ',len(true_labels))
    print('pred_labels = ',len(pred_labels))
    # Calculate F1-score for training data
    train_f1 = f1_score(true_labels, pred_labels)
    train_f1_scores.append(train_f1)
print(('f1 scores: ',train_f1_scores))
print('train_f1 = ',sum(train_f1_scores)/len(train_f1_scores))


## TEST
# test_f1_scores = []

# loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# model = model.to(device)
# ids = []
# labels = []
# print("Testing...")
# true_labels = []
# pred_labels = []
# try :
#     for sample in tqdm(loader):
#         X, ID = sample
#         #ID = ID[0]
#         X = X.to(device)
#         label_pred = model(X)
#         ids.extend(list(ID))
#         pred = (label_pred > 0.5).long()
#         pred = pred.cpu().detach().numpy().tolist()
#         labels.extend(pred)
# except:
#     print('ID = ',ID)

# dataresults = pd.DataFrame({'pred':pred,'labels')
# dataresults.to_csv('dataresults.csv')
# # Calculate F1-score for test data
# test_f1 = f1_score(labels[:len(pred)], pred)
# test_f1_scores.append(test_f1)
# print('train_f1_score = ',np.mean(train_f1_scores))
# print('test f1 score = ',test_f1)


# ### ENREGISTREMENT
# print("Saving...")
# tests = ["id,label\n"] + [f"{ID},{label_pred[0]}\n" for ID, label_pred in zip(ids, labels)]
# with open("submission.csv", "w") as file:
#     file.writelines(tests)


# Calculate average accuracy
# average_train_f1 = np.mean(train_f1_scores)
# average_test_f1 = np.mean(test_f1_scores)

# # Create the plot
# plt.figure(figsize=(8, 6))
# plt.plot(range(epochs), train_f1_scores, label='Training F1-score')
# plt.plot(range(len(test_f1_scores)), test_f1_scores, label='Test F1-score')
# plt.xlabel('Epochs')
# plt.ylabel('F1-score')
# plt.title(f'Training and Test F1-score\n'
#           f'Average Training F1: {average_train_f1:.3f}, Average Test F1: {average_test_f1:.3f}')
# plt.legend()

# # Save the plot to a file
# plt.savefig('f1_score_plot.png')

# # Display the plot
# plt.show()
