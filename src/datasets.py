"""
MV-TransUNet Dataset Pipeline

Retinal Vessel Segmentation Framework

Supported datasets:
    - DRIVE
    - STARE
    - CHASE_DB1
    - HRF


Features:
    - Automatic image-mask matching
    - CLAHE enhancement
    - Albumentations augmentation
    - PyTorch Dataset
    - Colab optimized DataLoader
    - Cross dataset evaluation support

"""


import os

import random

from pathlib import Path


import cv2

import numpy as np


import torch


from torch.utils.data import Dataset

from torch.utils.data import DataLoader

from torch.utils.data import random_split



import albumentations as A

from albumentations.pytorch import ToTensorV2





# ============================================================
# RANDOM SEED
# ============================================================


def set_seed(seed=42):


    random.seed(seed)


    np.random.seed(seed)


    torch.manual_seed(seed)


    torch.cuda.manual_seed(seed)


    torch.cuda.manual_seed_all(seed)





# ============================================================
# SUPPORTED IMAGE FORMATS
# ============================================================


IMAGE_EXTENSIONS = [

    ".png",

    ".jpg",

    ".jpeg",

    ".tif",

    ".tiff",

    ".ppm",

    ".gif"

]





# ============================================================
# CLAHE ENHANCEMENT
# ============================================================



class CLAHEEnhancement:


    def __init__(

        self,

        clip_limit=2.0,

        tile_grid_size=8

    ):


        self.clahe = cv2.createCLAHE(

            clipLimit=clip_limit,

            tileGridSize=(

                tile_grid_size,

                tile_grid_size

            )

        )




    def __call__(self,image):


        lab = cv2.cvtColor(

            image,

            cv2.COLOR_RGB2LAB

        )



        l_channel,a_channel,b_channel = cv2.split(

            lab

        )



        enhanced_l = self.clahe.apply(

            l_channel

        )



        merged = cv2.merge(

            [

                enhanced_l,

                a_channel,

                b_channel

            ]

        )



        output = cv2.cvtColor(

            merged,

            cv2.COLOR_LAB2RGB

        )


        return output





# ============================================================
# AUGMENTATIONS
# ============================================================



def get_train_transform(

    image_size=256

):


    return A.Compose(

        [

            A.Resize(

                height=image_size,

                width=image_size

            ),



            A.HorizontalFlip(

                p=0.5

            ),



            A.VerticalFlip(

                p=0.5

            ),



            A.Rotate(

                limit=30,

                p=0.5

            ),



            A.ElasticTransform(

                alpha=120,

                sigma=6,

                p=0.3

            ),



            A.RandomBrightnessContrast(

                p=0.5

            ),



            A.Normalize(

                mean=(

                    0.485,

                    0.456,

                    0.406

                ),

                std=(

                    0.229,

                    0.224,

                    0.225

                )

            ),



            ToTensorV2()

        ]

    )





def get_validation_transform(

    image_size=256

):


    return A.Compose(

        [

            A.Resize(

                height=image_size,

                width=image_size

            ),



            A.Normalize(

                mean=(

                    0.485,

                    0.456,

                    0.406

                ),

                std=(

                    0.229,

                    0.224,

                    0.225

                )

            ),



            ToTensorV2()

        ]

    )





# ============================================================
# FILE MATCHING
# ============================================================



def get_files(directory):


    directory = Path(directory)



    if not directory.exists():

        raise FileNotFoundError(

            f"Directory not found: {directory}"

        )



    files=[]



    for file in directory.iterdir():


        if file.suffix.lower() in IMAGE_EXTENSIONS:

            files.append(file)



    return sorted(files)





def normalize_filename(filename):


    name = Path(filename).stem.lower()



    replacements = [

        "_training",

        "_test",

        "_manual1",

        "_manual",

        "_1stho",

        "_2ndho",

        ".ah",

        ".vk"

    ]



    for item in replacements:


        name = name.replace(

            item,

            ""

        )



    return name





def match_image_masks(

    image_files,

    mask_files

):


    mask_dictionary={}



    for mask in mask_files:


        key = normalize_filename(

            mask.name

        )


        mask_dictionary[key]=mask




    pairs=[]



    for image in image_files:


        key = normalize_filename(

            image.name

        )



        if key in mask_dictionary:


            pairs.append(

                (

                    image,

                    mask_dictionary[key]

                )

            )



    if len(pairs)==0:


        raise RuntimeError(

            "No matching image-mask pairs found"

        )


    return pairs





# ============================================================
# RETINAL DATASET
# ============================================================



class RetinalVesselDataset(Dataset):


    def __init__(

        self,

        image_dir,

        mask_dir,

        transform=None,

        clahe=True

    ):


        super().__init__()



        self.image_dir = Path(image_dir)


        self.mask_dir = Path(mask_dir)



        self.transform = transform



        images = get_files(

            self.image_dir

        )



        masks = get_files(

            self.mask_dir

        )



        self.samples = match_image_masks(

            images,

            masks

        )



        self.clahe = None



        if clahe:


            self.clahe = CLAHEEnhancement()





    def __len__(self):


        return len(self.samples)





    def __getitem__(self,index):


        image_path,mask_path = self.samples[index]



        image = cv2.imread(

            str(image_path)

        )



        image = cv2.cvtColor(

            image,

            cv2.COLOR_BGR2RGB

        )



        mask = cv2.imread(

            str(mask_path),

            cv2.IMREAD_GRAYSCALE

        )



        if image is None:

            raise RuntimeError(

                f"Cannot read {image_path}"

            )


        if mask is None:

            raise RuntimeError(

                f"Cannot read {mask_path}"

            )



        if self.clahe is not None:


            image=self.clahe(

                image

            )



        mask=np.where(

            mask>127,

            1.0,

            0.0

        ).astype(

            np.float32

        )



        if self.transform:


            transformed=self.transform(

                image=image,

                mask=mask

            )


            image=transformed["image"]


            mask=transformed["mask"]



        else:


            image=torch.from_numpy(

                image.transpose(

                    2,

                    0,

                    1

                )

            ).float()



            mask=torch.from_numpy(

                mask

            )



        if mask.ndimension()==2:


            mask=mask.unsqueeze(

                0

            )



        return {


            "image":image,


            "mask":mask.float(),


            "image_path":str(image_path)


        }
    # ============================================================
# DATASET SPLITTING
# ============================================================


def create_train_validation_split(

    dataset,

    validation_ratio=0.2,

    seed=42

):


    validation_size = int(

        len(dataset)

        *

        validation_ratio

    )


    training_size = (

        len(dataset)

        -

        validation_size

    )


    generator = torch.Generator()


    generator.manual_seed(

        seed

    )


    train_dataset, validation_dataset = random_split(

        dataset,

        [

            training_size,

            validation_size

        ],

        generator=generator

    )


    return train_dataset, validation_dataset





# ============================================================
# DATALOADER CREATION
# ============================================================


def create_dataloader(

    dataset,

    batch_size=8,

    shuffle=True,

    num_workers=2,

    drop_last=False

):


    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=shuffle,

        num_workers=num_workers,

        pin_memory=True,

        persistent_workers=(num_workers > 0),

        drop_last=drop_last

    )


    return loader





# ============================================================
# TRAIN + VALIDATION PIPELINE
# ============================================================


def build_dataloaders(

    image_dir,

    mask_dir,

    image_size=256,

    batch_size=8,

    validation_ratio=0.2,

    num_workers=2,

    clahe=True

):


    full_dataset = RetinalVesselDataset(

        image_dir=image_dir,

        mask_dir=mask_dir,

        transform=get_train_transform(

            image_size

        ),

        clahe=clahe

    )



    train_dataset, validation_dataset = create_train_validation_split(

        full_dataset,

        validation_ratio

    )



    validation_dataset.dataset.transform = get_validation_transform(

        image_size

    )



    train_loader=create_dataloader(

        train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers

    )



    validation_loader=create_dataloader(

        validation_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers

    )



    return train_loader, validation_loader





# ============================================================
# TEST / EVALUATION DATASET
# ============================================================


def build_test_loader(

    image_dir,

    mask_dir,

    image_size=256,

    batch_size=1,

    num_workers=2,

    clahe=True

):


    dataset = RetinalVesselDataset(

        image_dir=image_dir,

        mask_dir=mask_dir,

        transform=get_validation_transform(

            image_size

        ),

        clahe=clahe

    )



    loader=create_dataloader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers

    )


    return loader





# ============================================================
# CROSS DATASET LOADERS
# ============================================================


def build_cross_dataset_loaders(

    datasets_config,

    image_size=256,

    batch_size=1,

    num_workers=2

):


    loaders={}



    for dataset in datasets_config:


        name = dataset["name"]



        print(

            f"Loading evaluation dataset: {name}"

        )



        loader = build_test_loader(

            image_dir=dataset["image_dir"],

            mask_dir=dataset["mask_dir"],

            image_size=image_size,

            batch_size=batch_size,

            num_workers=num_workers

        )



        loaders[name]=loader



    return loaders





# ============================================================
# DATASET STATISTICS
# ============================================================


def dataset_statistics(

    image_dir,

    mask_dir

):


    dataset = RetinalVesselDataset(

        image_dir=image_dir,

        mask_dir=mask_dir,

        transform=None,

        clahe=False

    )


    print(

        "==================================="

    )


    print(

        "Dataset Statistics"

    )


    print(

        "==================================="

    )


    print(

        "Images:",

        len(dataset)

    )


    print(

        "Image directory:",

        image_dir

    )


    print(

        "Mask directory:",

        mask_dir

    )


    print(

        "==================================="

    )





# ============================================================
# DATASET VERIFICATION
# ============================================================


def verify_dataset(

    image_dir,

    mask_dir

):


    print(

        "\nChecking dataset..."

    )


    try:


        dataset=RetinalVesselDataset(

            image_dir=image_dir,

            mask_dir=mask_dir,

            transform=None

        )



        print(

            "Dataset OK"

        )


        print(

            "Number of samples:",

            len(dataset)

        )



        sample=dataset[0]



        print(

            "Image shape:",

            sample["image"].shape

        )


        print(

            "Mask shape:",

            sample["mask"].shape

        )


        return True



    except Exception as error:


        print(

            "Dataset Error:"

        )


        print(

            error

        )


        return False





# ============================================================
# QUICK TEST
# ============================================================


if __name__ == "__main__":


    print(

        "MV-TransUNet Dataset Pipeline Test"

    )


    test_image="./datasets_processed/DRIVE/images"


    test_mask="./datasets_processed/DRIVE/masks"



    if os.path.exists(test_image) and os.path.exists(test_mask):


        verify_dataset(

            test_image,

            test_mask

        )


    else:


        print(

            "Dataset path does not exist."

        )


        print(

            "Run prepare_datasets.py first."

        )