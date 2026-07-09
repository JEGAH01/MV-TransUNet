"""
MV-TransUNet Training Engine

Colab GPU Optimized

Features:

- YAML configuration
- Mixed Precision Training
- AdamW optimizer
- Cosine scheduler
- Gradient accumulation
- Gradient clipping
- TensorBoard logging
- Best model saving
- Last checkpoint saving
- Resume training
- Early stopping
- Validation Dice

"""


import os
import random
import yaml


import numpy as np


from tqdm import tqdm


import torch


from torch.amp import autocast, GradScaler


from torch.optim import AdamW

from torch.optim.lr_scheduler import CosineAnnealingLR


from torch.utils.tensorboard import SummaryWriter



from models.mv_transunet import MVTransUNet

from models.losses import MVTransUNetLoss


from src.datasets import build_dataloaders





# ============================================================
# RANDOM SEED
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

    with open(path, "r") as file:

        return yaml.safe_load(file)





# ============================================================
# DICE SCORE
# ============================================================


def dice_score(prediction, target, threshold=0.5):


    prediction = torch.sigmoid(
        prediction
    )


    prediction = (
        prediction > threshold
    ).float()



    intersection = torch.sum(
        prediction * target
    )



    dice = (

        2.0 * intersection + 1e-6

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



    for step, batch in enumerate(progress):


        images = batch["image"].to(

            device,

            non_blocking=True

        )


        masks = batch["mask"].to(

            device,

            non_blocking=True

        )



        with autocast(

            device_type="cuda",

            enabled=device.type == "cuda"

        ):


            outputs = model(images)



            loss_dict = criterion(

                outputs,

                masks

            )


            loss = loss_dict["total_loss"]



            loss = loss / accumulation_steps





        scaler.scale(loss).backward()



        # Update weights after accumulation

        if (

            (step + 1) % accumulation_steps == 0

            or

            step == len(loader)-1

        ):



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



    total_loss = 0.0

    total_dice = 0.0



    with torch.no_grad():


        for batch in tqdm(

            loader,

            desc="Validation"

        ):



            images = batch["image"].to(

                device

            )



            masks = batch["mask"].to(

                device

            )



            outputs = model(images)



            loss_dict = criterion(

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
# SAVE CHECKPOINT
# ============================================================


def save_checkpoint(

    model,

    optimizer,

    scheduler,

    scaler,

    epoch,

    dice,

    path

):


    torch.save(

        {


            "epoch":

                epoch,


            "model_state":

                model.state_dict(),


            "optimizer_state":

                optimizer.state_dict(),


            "scheduler_state":

                scheduler.state_dict(),


            "scaler_state":

                scaler.state_dict(),


            "dice":

                dice


        },

        path

    )







# ============================================================
# LOAD CHECKPOINT
# ============================================================


def load_checkpoint(

    path,

    model,

    optimizer,

    scheduler,

    scaler,

    device

):


    checkpoint = torch.load(

        path,

        map_location=device

    )



    model.load_state_dict(

        checkpoint["model_state"]

    )


    optimizer.load_state_dict(

        checkpoint["optimizer_state"]

    )


    scheduler.load_state_dict(

        checkpoint["scheduler_state"]

    )


    scaler.load_state_dict(

        checkpoint["scaler_state"]

    )



    return checkpoint["epoch"], checkpoint["dice"]







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



    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )



    print("="*60)

    print(

        "Device:",

        device

    )

    print("="*60)



    if device.type == "cuda":

        print(

            torch.cuda.get_device_name(0)

        )





    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------


    train_loader, val_loader = build_dataloaders(


        image_dir=

        config["dataset"]["train_dataset"]["image_dir"],



        mask_dir=

        config["dataset"]["train_dataset"]["mask_dir"],



        image_size=

        config["preprocessing"]["image_size"]["height"],



        batch_size=

        config["dataloader"]["batch_size"],



        num_workers=

        config["dataloader"]["num_workers"]

    )





    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------


    model = MVTransUNet(

        pretrained=True

    ).to(device)





    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------


    criterion = MVTransUNetLoss().to(device)





    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------


    optimizer = AdamW(

        model.parameters(),

        lr=config["optimizer"]["learning_rate"],

        weight_decay=config["optimizer"]["weight_decay"]

    )





    scheduler = CosineAnnealingLR(

        optimizer,

        T_max=config["training"]["epochs"]

    )





    scaler = GradScaler(

        "cuda",

        enabled=device.type=="cuda"

    )





    # --------------------------------------------------------
    # LOGGING
    # --------------------------------------------------------


    os.makedirs(

        config["checkpoint"]["directory"],

        exist_ok=True

    )



    writer = SummaryWriter(

        config["logging"]["directory"]

    )





    best_dice = 0.0

    patience = 0



    start_epoch = 0





    # --------------------------------------------------------
    # RESUME
    # --------------------------------------------------------


    resume_path = os.path.join(

        config["checkpoint"]["directory"],

        "last_model.pth"

    )



    if os.path.exists(resume_path):


        print(

            "Resuming checkpoint..."

        )


        start_epoch, best_dice = load_checkpoint(

            resume_path,

            model,

            optimizer,

            scheduler,

            scaler,

            device

        )



        print(

            "Resume epoch:",

            start_epoch

        )





    # --------------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------------


    epochs = config["training"]["epochs"]



    for epoch in range(

        start_epoch,

        epochs

    ):



        print()

        print(

            f"Epoch {epoch+1}/{epochs}"

        )



        train_loss = train_one_epoch(

            model,

            train_loader,

            criterion,

            optimizer,

            scaler,

            device,

            config["training"]["gradient_accumulation_steps"]

        )



        val_loss, val_dice = validate(

            model,

            val_loader,

            criterion,

            device

        )



        scheduler.step()



        lr = optimizer.param_groups[0]["lr"]



        print(

            "Train Loss:",

            train_loss

        )


        print(

            "Validation Loss:",

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





        save_checkpoint(

            model,

            optimizer,

            scheduler,

            scaler,

            epoch,

            val_dice,

            os.path.join(

                config["checkpoint"]["directory"],

                "last_model.pth"

            )

        )





        if val_dice > best_dice:


            best_dice = val_dice

            patience = 0



            save_checkpoint(

                model,

                optimizer,

                scheduler,

                scaler,

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


            patience += 1



        if patience >= config["training"]["early_stopping"]["patience"]:


            print(

                "Early stopping triggered"

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





if __name__ == "__main__":

    main()