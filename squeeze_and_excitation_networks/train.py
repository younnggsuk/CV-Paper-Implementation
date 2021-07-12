import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from models import se_resnet
from models import evaluate, accuracy
from datasets import fetch_dataloader


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="data", type=str,
                    help="Directory containing the dataset")

parser.add_argument("--save_dir", default="experiments", type=str,
                    help="Directory containing model checkpoints")

parser.add_argument("--tensorboard_dir", default="runs", type=str,
                    help="Directory containing tensorboard events")

parser.add_argument("--num_layers", default=50, type=int,
                    help="Number of layers on SE-ResNet")

parser.add_argument("--epochs", default=100, type=int,
                    help="Number of epochs")

parser.add_argument("--batch_size", default=32, type=int,
                    help="Number of mini-batch size")

parser.add_argument("--learning_rate", default=0.6, type=float,
                    help="Initial learning rate")


if __name__ == "__main__":
    args = parser.parse_args()
    # data directory path check
    assert os.path.exists(args.data_dir), "Data directory not exist!"

    # save directory path check
    if not os.path.exists(args.save_dir):
        os.mkdir("experiments")

    # layer number check
    assert args.num_layers in (50, 101, 152), "Invalid layer number!"

    torch.manual_seed(11)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(11)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    dataloaders = fetch_dataloader(["train", "val"],
                                   data_dir = args.data_dir,
                                   batch_size=int(args.batch_size),
                                   num_workers=2)
    train_loader, val_loader = dataloaders["train"], dataloaders["val"]

    model = se_resnet(args.num_layers)
    model.fc = nn.Linear(model.fc.in_features, 2)
    nn.init.constant_(model.fc.bias, 0)
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(),
                    lr=args.learning_rate,
                    momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer,
                                    step_size=30,
                                    gamma=0.1)
    
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, 
                                        f"se_resnet_{args.num_layers}"))
    best_val_loss = float("inf") 

    for epoch in range(args.epochs):

        train_loss = 0
        train_acc = 0

        model.train()

        for iteration, (images, labels) in tqdm(enumerate(train_loader, start=1),
                                                desc="train loss, acc",
                                                total=len(train_loader)):

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                acc = accuracy(outputs, labels)

            train_loss += loss.item()
            train_acc += acc.item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss = evaluate(val_loader, model, loss_fn, device, desc="validation loss")
        val_acc = evaluate(val_loader, model, accuracy, device, desc="validation acc ")

        writer.add_scalars("Loss",
                           {"train": train_loss, "valid": val_loss},
                           epoch + 1)
        writer.add_scalars("Accuracy",
                           {"train": train_acc, "valid": val_acc},
                           epoch + 1)

        tqdm.write(f"Epoch {epoch + 1}/{args.epochs}\n"
                   f"[Train] loss: {train_loss:.3f}, acc: {train_acc:.3f}\n"
                   f"[Valid] loss: {val_loss:.3f}, acc: {val_acc:.3f}")

        scheduler.step(val_loss)

        if (val_loss < best_val_loss):
            torch.save({"epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()},
                       os.path.join(args.save_dir, f"se_resnet_{args.num_layers}_best.tar"))
            best_val_loss = val_loss
            print(f"Saved best model - val_loss: {best_val_loss:.3f}")
    
    writer.close()