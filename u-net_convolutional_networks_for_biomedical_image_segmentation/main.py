import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from models import unet
from models import Evaluator, train, validation
from datasets import fetch_dataloader


def str2bool(arg):
    if isinstance(arg, bool):
        return arg
    elif arg.lower() in ("t", "true", "y", "yes", "1"):
        return True
    elif arg.lower() in ("f", "false", "n", "no", "0"):
        return False
    else:
        return None


parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", default="data", type=str,
                    help="Directory containing the dataset")

parser.add_argument("--save_dir", default="experiments", type=str,
                    help="Directory containing model checkpoints")

parser.add_argument("--tensorboard_dir", default="runs", type=str,
                    help="Directory containing tensorboard events")

parser.add_argument("--use_batchnorm", default=True, type=str2bool,
                    help="Whether to use Batch Normalization [Y/N]")

parser.add_argument("--in_channels", default=3, type=int,
                    help="Number of input channels")

parser.add_argument("--num_classes", default=21, type=int,
                    help="Number of classes")

parser.add_argument("--epochs", default=100, type=int,
                    help="Number of epochs")

parser.add_argument("--batch_size", default=1, type=int,
                    help="Number of mini-batch size")

parser.add_argument("--learning_rate", default=0.1, type=float,
                    help="Initial learning rate")


if __name__ == "__main__":
    args = parser.parse_args()
    # data directory path check
    assert os.path.exists(args.data_dir), "Data directory does not exist!"

    # save directory path check
    if not os.path.exists(args.save_dir):
        os.mkdir("experiments")

    torch.manual_seed(11)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(11)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    dataloaders = fetch_dataloader(["train", "val", "eval"],
                                   data_dir=args.data_dir,
                                   batch_size=args.batch_size,
                                   num_workers=2)
    train_loader, val_loader, eval_loader = dataloaders["train"], dataloaders["val"], dataloaders["eval"]

    model = unet(args.in_channels, args.num_classes, bool(args.use_batchnorm))
    model = model.to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                               factor=0.1,
                                               patience=5)
    
    writer = SummaryWriter(os.path.join(args.tensorboard_dir, "unet"))
    evaluator = Evaluator(args.num_classes)
    best_mean_iou = 0

    for epoch in range(args.epochs):
        # train
        train_loss = train(train_loader, model, loss_fn, optimizer, device)

        # validation
        val_loss = validation(val_loader, model, loss_fn, device)

        # evaluate miou, pixel accuracy
        evaluator.reset()
        evaluator.evaluate(eval_loader, model, device)
        iou, pixel_acc, mean_iou, mean_acc = evaluator.get_metrics()

        print((f"Epoch {epoch + 1}/{args.epochs}\n"
               f"[Train] loss: {train_loss:.3f}\n"
               f"[Valid] loss: {val_loss:.3f}, mIoU: {mean_iou:.3f}, Mean Accuracy: {mean_acc:.3f}"))

        writer.add_scalars("Loss", {"train": train_loss, "valid": val_loss}, epoch + 1)
        writer.add_scalar("Mean IoU", mean_iou, epoch + 1)
        writer.add_scalar("Mean Accuracy", mean_acc, epoch + 1)
        writer.add_scalars("IoU", 
            {"0. background": iou[0],
            "1. aeroplane":  iou[1],
            "2. bicycle":    iou[2],
            "3. bird":       iou[3],
            "4. boat":       iou[4],
            "5. bottle":     iou[5],
            "6. bus":        iou[6],
            "7. car":        iou[7],
            "8. cat":        iou[8],
            "9. chair":      iou[9],
            "10. cow":       iou[10],
            "11. diningtable": iou[11],
            "12. dog":       iou[12],
            "13. horse":     iou[13],
            "14. motorbike": iou[14],
            "15. person":    iou[15],
            "16. potted plant": iou[16],
            "17. sheep":     iou[17],
            "18. sofa":      iou[18],
            "19. train":     iou[19],
            "20. tv/monitor": iou[20]}, epoch + 1)
        writer.add_scalars("Pixel_Accuracy", 
            {"0. background": pixel_acc[0],
            "1. aeroplane":   pixel_acc[1],
            "2. bicycle":     pixel_acc[2],
            "3. bird":        pixel_acc[3],
            "4. boat":        pixel_acc[4],
            "5. bottle":      pixel_acc[5],
            "6. bus":         pixel_acc[6],
            "7. car":         pixel_acc[7],
            "8. cat":         pixel_acc[8],
            "9. chair":       pixel_acc[9],
            "10. cow":        pixel_acc[10],
            "11. diningtable": pixel_acc[11],
            "12. dog":        pixel_acc[12],
            "13. horse":      pixel_acc[13],
            "14. motorbike":  pixel_acc[14],
            "15. person":     pixel_acc[15],
            "16. potted plant": pixel_acc[16],
            "17. sheep":      pixel_acc[17],
            "18. sofa":       pixel_acc[18],
            "19. train":      pixel_acc[19],
            "20. tv/monitor": pixel_acc[20]}, epoch + 1)

        if (mean_iou > best_mean_iou):
            torch.save({"epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict()},
                        os.path.join(args.save_dir, f"unet_best_miou_{mean_iou:.3f}.tar"))
            best_mean_iou = mean_iou
            print(f"Saved best model - best_miou: {best_mean_iou:.3f}")

        scheduler.step(val_loss)
        
    writer.close()