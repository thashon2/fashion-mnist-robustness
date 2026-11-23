"""Fashion attributes: small independent CNN baselines and a shared multi-task CNN.

Run in Google Colab after downloading the Kaggle dataset. See README.md.
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
PATTERNS = {
    'plain': r'\b(solid|plain)\b',
    'striped': r'\b(striped|stripes|stripe)\b',
    'checked': r'\b(checkered|checked|checks|check|plaid)\b',
    'floral': r'\b(floral|flower print|flowered)\b',
}
TASKS = ('category', 'colour', 'pattern')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pattern_from_text(value):
    """Only assign a pattern when one unambiguous keyword is present."""
    text = str(value).lower()
    matches = [name for name, regex in PATTERNS.items() if re.search(regex, text)]
    return matches[0] if len(matches) == 1 else None


def find_dataset(root):
    csvs = list(root.rglob('styles.csv'))
    if not csvs:
        raise FileNotFoundError('styles.csv not found beneath --data-dir')
    csv_path = csvs[0]
    dirs = [x for x in root.rglob('images') if x.is_dir()]
    if not dirs:
        raise FileNotFoundError('images/ not found beneath --data-dir')
    return csv_path, dirs[0]


def prepare_rows(root, max_images, seed, min_class_size):
    csv_path, images_dir = find_dataset(root)
    frame = pd.read_csv(csv_path, on_bad_lines='skip')
    needed = ['id', 'masterCategory', 'articleType', 'baseColour', 'productDisplayName']
    if not set(needed).issubset(frame.columns):
        raise ValueError(f'Metadata missing required fields: {set(needed)-set(frame.columns)}')
    frame = frame[frame['masterCategory'].astype(str).str.lower() == 'apparel'].copy()
    frame = frame.dropna(subset=['id','articleType','baseColour'])
    frame['image_path'] = frame['id'].apply(lambda v: str(images_dir / f'{int(v)}.jpg'))
    frame = frame[frame['image_path'].map(lambda s: Path(s).exists())].copy()
    frame['category'] = frame['articleType'].astype(str).str.strip()
    frame['colour'] = frame['baseColour'].astype(str).str.strip()
    frame['pattern'] = frame['productDisplayName'].map(pattern_from_text)
    frame = frame.drop_duplicates('id')
    # Reduce the number of rare labels, instead of pretending those classes have enough data.
    for key in ('category','colour'):
        counts = frame[key].value_counts()
        frame = frame[frame[key].isin(counts[counts >= min_class_size].index)]
    if max_images and len(frame) > max_images:
        frame = frame.sample(n=max_images, random_state=seed)
    frame = frame.reset_index(drop=True)
    if len(frame) < 30:
        raise ValueError('Too few valid clothing images; check dataset path and filtering')
    # Keep only pattern labels with enough examples; others become missing (masked labels).
    counts = frame['pattern'].value_counts()
    valid_patterns = set(counts[counts >= max(8, min_class_size // 3)].index)
    frame.loc[~frame['pattern'].isin(valid_patterns), 'pattern'] = None
    return frame, csv_path, valid_patterns


def split_rows(frame, seed):
    # Fixed split, shared by every model; repeated image IDs are removed beforehand.
    cats = frame['category'].value_counts()
    stratify = frame['category'] if cats.min() >= 3 else None
    train, rem = train_test_split(frame, test_size=.30, random_state=seed, stratify=stratify)
    rem_counts = rem['category'].value_counts()
    rem_strat = rem['category'] if rem_counts.min() >= 2 else None
    valid, test = train_test_split(rem, test_size=.50, random_state=seed, stratify=rem_strat)
    return train.copy(), valid.copy(), test.copy()


class ClothesDataset(Dataset):
    def __init__(self, frame, maps, train=False):
        self.frame = frame.reset_index(drop=True)
        self.maps = maps
        aug = [transforms.Resize((128, 128))]
        if train:
            aug.append(transforms.RandomAffine(degrees=10, translate=(.10,.10)))
        aug += [transforms.ToTensor(), transforms.Normalize([.5]*3,[.5]*3)]
        self.transform = transforms.Compose(aug)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row['image_path']) as image:
            tensor = self.transform(image.convert('RGB'))
        labels = {}
        for task in TASKS:
            value = row[task]
            labels[task] = self.maps[task].get(str(value), -100) if pd.notna(value) else -100
        return tensor, {k:torch.tensor(v, dtype=torch.long) for k,v in labels.items()}


class FeatureCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        previous=3
        for channels in (32,64,128):
            layers.extend([nn.Conv2d(previous,channels,3,padding=1),nn.ReLU(),nn.MaxPool2d(2)])
            previous=channels
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.net=nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x).flatten(1)


class MultiTaskCNN(nn.Module):
    def __init__(self, counts, tasks):
        super().__init__()
        self.features=FeatureCNN()
        self.heads=nn.ModuleDict({task:nn.Linear(128,counts[task]) for task in tasks})

    def forward(self,x):
        features=self.features(x)
        return {k:head(features) for k,head in self.heads.items()}


def epoch_loop(model, loader, optimizer, device, tasks):
    training=optimizer is not None
    model.train(training)
    running_loss=0.0
    n_batches=0
    truth={t:[] for t in tasks}
    pred={t:[] for t in tasks}
    for x,labels in loader:
        x=x.to(device)
        labels={t:labels[t].to(device) for t in tasks}
        with torch.set_grad_enabled(training):
            outputs=model(x)
            losses=[]
            for t in tasks:
                use=labels[t] >= 0
                if use.any():
                    losses.append(nn.functional.cross_entropy(outputs[t][use],labels[t][use]))
                    truth[t].extend(labels[t][use].cpu().tolist())
                    pred[t].extend(outputs[t][use].argmax(1).detach().cpu().tolist())
            if not losses:
                continue
            loss=sum(losses)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        running_loss+=float(loss.item())
        n_batches+=1
    result={'loss':running_loss/max(n_batches,1)}
    for t in tasks:
        result[f'{t}_samples']=len(truth[t])
        result[f'{t}_accuracy']=accuracy_score(truth[t],pred[t]) if truth[t] else None
        result[f'{t}_macro_f1']=f1_score(truth[t],pred[t],average='macro',zero_division=0) if truth[t] else None
    return result,truth,pred


def save_plots(history, out, name):
    plt.figure(figsize=(6,4))
    plt.plot([x['epoch'] for x in history],[x['train_loss'] for x in history],label='Train')
    plt.plot([x['epoch'] for x in history],[x['valid_loss'] for x in history],label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out/f'{name}_learning_curve.png',dpi=160)
    plt.close()


def train_model(name, tasks, loaders, maps, epochs, lr, device, output):
    model=MultiTaskCNN({t:len(maps[t]) for t in tasks},tasks).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=lr)
    history=[]
    best=float('inf')
    path=output/f'{name}_best.pt'
    for epoch in range(1,epochs+1):
        train_metrics,_,_=epoch_loop(model,loaders['train'],optimizer,device,tasks)
        with torch.no_grad():
            valid_metrics,_,_=epoch_loop(model,loaders['valid'],None,device,tasks)
        history.append({'epoch':epoch,'train_loss':train_metrics['loss'],'valid_loss':valid_metrics['loss'],
                        'train':train_metrics,'valid':valid_metrics})
        print(f'{name}: epoch {epoch}/{epochs}, train loss={train_metrics["loss"]:.4f}, valid loss={valid_metrics["loss"]:.4f}')
        if valid_metrics['loss'] < best:
            best=valid_metrics['loss']
            torch.save({'weights':model.state_dict(),'tasks':tasks,'maps':maps,'epoch':epoch,'seed':args.seed},path)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=False)['weights'])
    with torch.no_grad():
        test_metrics,actual,pred=epoch_loop(model,loaders['test'],None,device,tasks)
    for task in tasks:
        pd.DataFrame(confusion_matrix(actual[task],pred[task],labels=range(len(maps[task])))).to_csv(
            output/f'{name}_{task}_confusion.csv',index=False)
    save_plots(history,output,name)
    with open(output/f'{name}_history.json','w') as file:
        json.dump(history,file,indent=2)
    return {'tasks':list(tasks),'parameters':sum(p.numel() for p in model.parameters()),
            'best_validation_loss':best,'test':test_metrics,'checkpoint':str(path)}


def main(args):
    seed_everything(args.seed)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    frame,csv_path,patterns=prepare_rows(Path(args.data_dir),args.max_images,args.seed,args.min_class_size)
    train,valid,test=split_rows(frame,args.seed)
    maps={task:{label:i for i,label in enumerate(sorted(train[task].dropna().astype(str).unique()))}
          for task in TASKS}
    # If an attribute is too rare, don't invent a result for it.
    tasks=[task for task in TASKS if len(maps[task])>=2]
    if 'pattern' not in tasks:
        print('WARNING: insufficient verified pattern labels. Pattern task not trained; report this limitation.')
    if len(tasks)<2:
        raise ValueError('Need at least two label groups with multiple classes')
    for part,subset in [('train',train),('valid',valid),('test',test)]:
        subset[['id','category','colour','pattern']].to_csv(out/f'{part}_split.csv',index=False)
    loaders={key:DataLoader(ClothesDataset(subset,maps,key=='train'),batch_size=args.batch_size,
                             shuffle=key=='train',num_workers=0)
             for key,subset in [('train',train),('valid',valid),('test',test)]}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}; train={len(train)}, valid={len(valid)}, test={len(test)}; tasks={tasks}')
    results={'data_source':str(csv_path),'split_counts':{'train':len(train),'valid':len(valid),'test':len(test)},
             'pattern_labeled_count':int(frame['pattern'].notna().sum()),'pattern_classes':sorted(patterns),
             'seed':args.seed,'device':str(device),'results':{}}
    for task in tasks:
        key=f'baseline_{task}'
        results['results'][key]=train_model(key,[task],loaders,maps,args.epochs,args.lr,device,out)
    results['results']['multi_task']=train_model('multi_task',tasks,loaders,maps,args.epochs,args.lr,device,out)
    with open(out/'metrics.json','w') as file:
        json.dump(results,file,indent=2)
    print('Saved actual metrics and learning curves in',out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-dir',required=True,help='Directory containing images/ and styles.csv')
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--max-images',type=int,default=3500,help='0 means all filtered images')
    parser.add_argument('--min-class-size',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    main(args)
