import torch
from tqdm import tqdm


__all__ = ['evaluate', 'accuracy']


def evaluate(data_loader, model, eval_fn, device, desc="steps"):
    model.eval()
    
    score = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc=desc, total=len(data_loader)):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            score += eval_fn(outputs, labels).item()
            
    return score / len(data_loader)


def accuracy(outputs, labels):
    y_preds = torch.argmax(outputs, dim=1)
    return (y_preds == labels).float().mean()
        