"""
MV-TransUNet Training Engine

Colab GPU Optimized

Features:

- Mixed Precision Training
- AdamW
- Cosine Scheduler
- Gradient Accumulation
- Gradient Clipping
- TensorBoard
- Best Model Saving
- Last Checkpoint Saving
- Early Stopping
- Validation Dice

"""


import os
import random
import yaml

import numpy as np

from tqdm import tqdm


import torch
import torch.nn as nn


from torch.amp import autocast, GradScaler


from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.utils.tensorboard import SummaryWriter



from models.mv_transunet import MVTransUNet

from models.losses import MVTransUNetLoss


from src.datasets import build_dataloaders





# ============================================================
# SEED
# ============================================================


def seed_everything(seed):


    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)






# ============================================================
# CONFIG
# ============================================================


def load_config(path):


    with open(path,"r") as file:

        return yaml.safe_load(file)







# ============================================================
# DICE
# ============================================================


def dice_score(

        prediction,

        target,

        threshold=0.5

):


    prediction = torch.sigmoid(prediction)



    prediction = (

        prediction > threshold

    ).float()



    intersection = torch.sum(

        prediction * target

    )



    dice = (

        2 * intersection + 1e-6

    ) / (

        torch.sum(prediction)

        +

        torch.sum(target)

        +

        1e-6

    )


    return dice







# ============================================================
# TRAIN ONE EPOCH
# ============================================================


def train_one_epoch(

        model,

        loader,

        criterion,

        optimizer,

        scaler,

        device,

        accumulation_steps

):


    model.train()


    running_loss = 0.0



    optimizer.zero_grad()



    progress = tqdm(

        loader,

        desc="Training"

    )



    for step,batch in enumerate(progress):


        images = batch["image"].to(

            device,

            non_blocking=True

        )


        masks = batch["mask"].to(

            device,

            non_blocking=True

        )



        with autocast(

            device_type=device.type,

            enabled=device.type=="cuda"

        ):


            outputs = model(images)


            loss_dict = criterion(

                outputs,

                masks

            )


            loss = loss_dict["total_loss"]



            loss = loss / accumulation_steps



        scaler.scale(loss).backward()



        if (

            step + 1

        ) % accumulation_steps == 0:



            scaler.unscale_(optimizer)



            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0

            )



            scaler.step(

                optimizer

            )


            scaler.update()



            optimizer.zero_grad()



        running_loss += loss.item()



        progress.set_postfix(

            loss=loss.item()

        )



    return running_loss / len(loader)








# ============================================================
# VALIDATION
# ============================================================


def validate(

        model,

        loader,

        criterion,

        device

):


    model.eval()


    total_loss = 0

    total_dice = 0



    with torch.no_grad():


        for batch in tqdm(

            loader,

            desc="Validation"

        ):


            images=batch["image"].to(

                device

            )


            masks=batch["mask"].to(

                device

            )



            outputs=model(images)



            loss_dict=criterion(

                outputs,

                masks

            )



            total_loss += loss_dict["total_loss"].item()



            total_dice += dice_score(

                outputs,

                masks

            ).item()



    return (

        total_loss / len(loader),

        total_dice / len(loader)

    )








# ============================================================
# CHECKPOINT
# ============================================================


def save_checkpoint(

        model,

        optimizer,

        scheduler,

        epoch,

        dice,

        path

):


    torch.save(

        {


            "epoch":epoch,


            "model_state":

                model.state_dict(),


            "optimizer_state":

                optimizer.state_dict(),


            "scheduler_state":

                scheduler.state_dict(),


            "dice":

                dice


        },

        path

    )








# ============================================================
# MAIN
# ============================================================


def main():



    config = load_config(

        "config.yaml"

    )



    seed_everything(

        config["seed"]["value"]

    )



    device=torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )



    print()

    print("==============================")

    print(

        "Device:",

        device

    )

    print("==============================")





    if device.type=="cuda":


        print(

            torch.cuda.get_device_name(0)

        )






    # ========================================================
    # DATA
    # ========================================================



    train_loader,val_loader = build_dataloaders(



        image_dir=

        "./datasets_processed/DRIVE/images",



        mask_dir=

        "./datasets_processed/DRIVE/masks",



        image_size=

        config["preprocessing"]["image_size"]["height"],



        batch_size=

        config["dataloader"]["batch_size"],



        num_workers=

        config["dataloader"]["num_workers"]

    )







    # ========================================================
    # MODEL
    # ========================================================



    model=MVTransUNet(

        pretrained=True

    )



    model=model.to(device)







    # ========================================================
    # LOSS
    # ========================================================



    criterion=MVTransUNetLoss()







    # ========================================================
    # OPTIMIZER
    # ========================================================



    optimizer=AdamW(

        model.parameters(),

        lr=config["optimizer"]["learning_rate"],

        weight_decay=config["optimizer"]["weight_decay"]

    )







    scheduler=CosineAnnealingLR(

        optimizer,

        T_max=config["training"]["epochs"]

    )







    scaler=GradScaler(

        "cuda",

        enabled=device.type=="cuda"

    )








    # ========================================================
    # LOGGING
    # ========================================================



    writer=SummaryWriter(

        config["logging"]["directory"]

    )



    os.makedirs(

        config["checkpoint"]["directory"],

        exist_ok=True

    )







    best_dice=0


    patience=0





    # ========================================================
    # TRAIN LOOP
    # ========================================================


    epochs=config["training"]["epochs"]



    for epoch in range(epochs):


        print()

        print(

            f"Epoch {epoch+1}/{epochs}"

        )



        train_loss=train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            scaler,

            device,

            config["training"]["gradient_accumulation_steps"]

        )



        val_loss,val_dice=validate(

            model,

            val_loader,

            criterion,

            device

        )



        scheduler.step()



        lr=optimizer.param_groups[0]["lr"]



        print(

            "Train Loss:",

            train_loss

        )


        print(

            "Val Loss:",

            val_loss

        )


        print(

            "Dice:",

            val_dice

        )


        print(

            "LR:",

            lr

        )






        writer.add_scalar(

            "Loss/train",

            train_loss,

            epoch

        )



        writer.add_scalar(

            "Loss/val",

            val_loss,

            epoch

        )



        writer.add_scalar(

            "Dice/val",

            val_dice,

            epoch

        )



        writer.add_scalar(

            "LR",

            lr,

            epoch

        )






        save_checkpoint(

            model,

            optimizer,

            scheduler,

            epoch,

            val_dice,

            os.path.join(

                config["checkpoint"]["directory"],

                "last_model.pth"

            )

        )






        if val_dice > best_dice:



            best_dice=val_dice


            patience=0



            save_checkpoint(

                model,

                optimizer,

                scheduler,

                epoch,

                val_dice,

                os.path.join(

                    config["checkpoint"]["directory"],

                    "best_model.pth"

                )

            )



            print(

                "Saved Best Model"

            )



        else:


            patience +=1



        if patience >= config["training"]["early_stopping"]["patience"]:


            print(

                "Early stopping"

            )


            break







    writer.close()



    print()

    print(

        "Training Complete"

    )

    print(

        "Best Dice:",

        best_dice

    )








if __name__=="__main__":

    main()