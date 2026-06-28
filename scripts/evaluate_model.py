#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score

from skin_dss.data.ham10000_dataset import HAM10000Dataset, get_eval_transforms


def build_model(arch: str, num_classes: int, device: str):
    if arch == 'efficientnet_b0':
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(weights=None)
        in_f = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(in_f, num_classes))
    elif arch == 'resnet18':
        from torchvision import models
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f'Unsupported arch: {arch}')

    model.to(device)
    return model



def evaluate(model, dataloader, device: str, num_classes: int):
    model.eval()
    preds = []
    probs = []
    targets = []
    softmax = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Eval'):
            images = images.to(device)
            labels = labels.to(device)
            out = model(images)
            p = softmax(out)
            prob = p.cpu().numpy()
            pred = np.argmax(prob, axis=1)
            preds.extend(pred.tolist())
            probs.extend(prob.tolist())
            targets.extend(labels.cpu().numpy().tolist())

    preds = np.array(preds)
    probs = np.array(probs)
    targets = np.array(targets)

    acc = accuracy_score(targets, preds)
    report = classification_report(targets, preds, output_dict=True)
    cm = confusion_matrix(targets, preds)

    try:
        k = 3
        topk_preds = np.argsort(probs, axis=1)[:, -k:]
        topk_hit = [targets[i] in topk_preds[i] for i in range(len(targets))]
        top3_acc = float(np.mean(topk_hit))
    except Exception:
        top3_acc = None

    try:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(targets, classes=list(range(num_classes)))
        auc = roc_auc_score(y_bin, probs, average='macro', multi_class='ovr')
    except Exception:
        auc = None

    return {
        'accuracy': acc,
        'top3_accuracy': top3_acc,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc_auc_macro': float(auc) if auc is not None else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--csv', default='data/processed/ham10000_splits.csv')
    parser.add_argument('--split', default='test')
    parser.add_argument('--arch', default='efficientnet_b0', choices=['efficientnet_b0', 'resnet18'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default='experiments/eval_report.json')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    checkpoint = Path(args.checkpoint)
    device = args.device

    transforms = get_eval_transforms(image_size=args.image_size)
    ds = HAM10000Dataset(csv_path, split=args.split, transform=transforms)
    import pandas as pd
    df = pd.read_csv(csv_path)
    num_classes = int(df['label'].nunique())

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(args.arch, num_classes=num_classes, device=device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    results = evaluate(model, loader, device=device, num_classes=num_classes)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('Evaluation finished. Report saved to', out_path)


if __name__ == '__main__':
    main()
