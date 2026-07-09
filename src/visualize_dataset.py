"""
MV-TransUNet Dataset Visualization

Checks:
- image/mask alignment
- vessel visibility
- dataset integrity

Compatible with:
datasets_processed/

"""



from pathlib import Path

import random

import cv2

import numpy as np

import matplotlib.pyplot as plt





# ============================================================
# PATHS
# ============================================================


DATASET_DIR = Path(

    "./datasets_processed"

)


OUTPUT_DIR = Path(

    "./experiments/dataset_preview"

)





# ============================================================
# FILE LOADING
# ============================================================


def get_files(folder):


    extensions = {

        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".ppm",
        ".gif"

    }



    folder = Path(folder)



    if not folder.exists():

        return []



    files=[]



    for file in folder.iterdir():


        if file.suffix.lower() in extensions:


            files.append(file)



    return sorted(files)





# ============================================================
# MATCH IMAGE AND MASK
# ============================================================


def create_pairs(dataset):


    image_folder = (

        DATASET_DIR

        /

        dataset

        /

        "images"

    )


    mask_folder = (

        DATASET_DIR

        /

        dataset

        /

        "masks"

    )



    images=get_files(

        image_folder

    )


    masks=get_files(

        mask_folder

    )



    print(

        dataset,

        "images:",

        len(images),

        "masks:",

        len(masks)

    )



    mask_dictionary={}



    for mask in masks:


        mask_dictionary[

            mask.stem

        ] = mask




    pairs=[]



    for image in images:


        key=image.stem



        if key in mask_dictionary:


            pairs.append(

                (

                    image,

                    mask_dictionary[key]

                )

            )



    return pairs





# ============================================================
# LOAD SAMPLE
# ============================================================


def load_sample(dataset):


    pairs=create_pairs(

        dataset

    )



    if len(pairs)==0:


        raise RuntimeError(

            f"No pairs found for {dataset}"

        )



    image_path,mask_path=random.choice(

        pairs

    )



    image=cv2.imread(

        str(image_path)

    )


    image=cv2.cvtColor(

        image,

        cv2.COLOR_BGR2RGB

    )



    mask=cv2.imread(

        str(mask_path),

        cv2.IMREAD_GRAYSCALE

    )



    mask=(

        mask > 127

    ).astype(

        np.uint8

    )



    return image,mask,image_path.name





# ============================================================
# OVERLAY
# ============================================================


def create_overlay(

    image,

    mask

):


    overlay=image.copy()



    green=np.zeros_like(

        image

    )


    green[:,:,1]=255



    overlay=np.where(

        mask[:,:,None]==1,

        green,

        overlay

    )



    result=cv2.addWeighted(

        image,

        0.6,

        overlay,

        0.4,

        0

    )



    return result





# ============================================================
# VISUALIZATION
# ============================================================


def visualize_dataset(dataset):


    image,mask,name=load_sample(

        dataset

    )



    overlay=create_overlay(

        image,

        mask

    )



    plt.figure(

        figsize=(15,5)

    )



    plt.subplot(

        1,

        3,

        1

    )


    plt.imshow(image)

    plt.title(

        dataset+" Original"

    )

    plt.axis(

        "off"

    )



    plt.subplot(

        1,

        3,

        2

    )


    plt.imshow(

        mask,

        cmap="gray"

    )


    plt.title(

        "Ground Truth"

    )


    plt.axis(

        "off"

    )



    plt.subplot(

        1,

        3,

        3

    )


    plt.imshow(

        overlay

    )


    plt.title(

        "Overlay"

    )


    plt.axis(

        "off"

    )



    plt.suptitle(

        name

    )



    OUTPUT_DIR.mkdir(

        parents=True,

        exist_ok=True

    )



    save_path=(

        OUTPUT_DIR

        /

        f"{dataset}_preview.png"

    )



    plt.savefig(

        save_path,

        dpi=300,

        bbox_inches="tight"

    )


    plt.close()



    print(

        dataset,

        "saved:",

        save_path

    )





# ============================================================
# MAIN
# ============================================================


def main():


    for dataset in [

        "DRIVE",

        "STARE",

        "CHASE_DB1",

        "HRF"

    ]:


        visualize_dataset(

            dataset

        )



    print(

        "\nVisualization complete."

    )





if __name__=="__main__":


    main()