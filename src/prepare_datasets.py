"""
MV-TransUNet Dataset Preparation Pipeline

Supported:
- DRIVE
- STARE
- CHASE_DB1
- HRF/HFR


Output:

datasets_processed/

    DRIVE/
        images/
        masks/

    STARE/
        images/
        masks/

    CHASE_DB1/
        images/
        masks/

    HRF/
        images/
        masks/
"""


from pathlib import Path

import shutil



# ============================================================
# PATHS
# ============================================================


RAW_DATA_DIR = Path("./datasets")

OUTPUT_DIR = Path("./datasets_processed")



IMAGE_EXTENSIONS = {

    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".ppm",
    ".gif"

}



# ============================================================
# FILE UTILITIES
# ============================================================


def create_folder(path):

    path.mkdir(

        parents=True,

        exist_ok=True

    )





def clear_dataset_folder(name):

    folder = OUTPUT_DIR / name


    if folder.exists():

        shutil.rmtree(folder)





def get_files(folder):


    folder = Path(folder)


    if not folder.exists():

        return []



    files=[]


    for file in folder.rglob("*"):


        if file.is_file():


            if file.suffix.lower() in IMAGE_EXTENSIONS:

                files.append(file)



    return sorted(files)







# ============================================================
# NORMALIZATION
# ============================================================


def filename_key(file):


    name = file.stem.lower()



    # -------------------------
    # DRIVE
    # -------------------------

    replacements = [

        "_training",

        "_test",

        "_manual1",

        "_manual",

        "_mask"

    ]


    for item in replacements:

        name = name.replace(

            item,

            ""

        )



    # -------------------------
    # STARE
    # -------------------------

    name = name.replace(

        ".ah",

        ""

    )


    name = name.replace(

        ".vk",

        ""

    )



    # -------------------------
    # CHASE
    # -------------------------

    name = name.replace(

        "_1stho",

        ""

    )


    name = name.replace(

        "_2ndho",

        ""

    )



    # remove spaces

    name = name.replace(

        " ",

        ""

    )



    return name.strip()





# ============================================================
# MATCHING
# ============================================================


def pair_images_masks(images, masks):


    mask_dictionary={}



    for mask in masks:


        key=filename_key(mask)


        mask_dictionary[key]=mask




    pairs=[]


    missing=[]



    for image in images:


        key=filename_key(image)



        if key in mask_dictionary:


            pairs.append(

                (

                    image,

                    mask_dictionary[key]

                )

            )

        else:


            missing.append(

                image.name

            )



    if missing:


        print(

            "Unmatched images:",

            missing[:5]

        )



    return pairs







# ============================================================
# COPY
# ============================================================


def copy_pairs(pairs,name):


    clear_dataset_folder(name)



    image_output = (

        OUTPUT_DIR

        /

        name

        /

        "images"

    )


    mask_output = (

        OUTPUT_DIR

        /

        name

        /

        "masks"

    )



    create_folder(image_output)

    create_folder(mask_output)



    counter=1



    for image,mask in pairs:


        shutil.copy(

            image,

            image_output /

            f"{counter:03d}{image.suffix}"

        )


        shutil.copy(

            mask,

            mask_output /

            f"{counter:03d}{mask.suffix}"

        )


        counter+=1



    return counter-1








# ============================================================
# DRIVE
# ============================================================


def prepare_drive():


    print("\nPreparing DRIVE")



    drive=RAW_DATA_DIR/"DRIVE"



    images=[]

    masks=[]



    for folder in [

        drive/"training",

        drive/"test"

    ]:


        images.extend(

            get_files(

                folder/"images"

            )

        )



        masks.extend(

            get_files(

                folder/"masks"

            )

        )



    print(

        "DRIVE images:",

        len(images)

    )


    print(

        "DRIVE masks:",

        len(masks)

    )



    pairs=pair_images_masks(

        images,

        masks

    )



    print(

        "DRIVE pairs:",

        len(pairs)

    )



    return copy_pairs(

        pairs,

        "DRIVE"

    )







# ============================================================
# STARE
# ============================================================


def prepare_stare():


    print("\nPreparing STARE")


    stare=RAW_DATA_DIR/"STARE"



    images=[]

    masks=[]



    for folder in [

        stare/"train",

        stare/"val"

    ]:


        images.extend(

            get_files(

                folder/"input"

            )

        )


        masks.extend(

            get_files(

                folder/"label"

            )

        )



    pairs=pair_images_masks(

        images,

        masks

    )


    print(

        "STARE pairs:",

        len(pairs)

    )


    return copy_pairs(

        pairs,

        "STARE"

    )







# ============================================================
# CHASE
# ============================================================


def prepare_chase():


    print("\nPreparing CHASE_DB1")



    chase=RAW_DATA_DIR/"CHASE_DB1"



    images=[]

    masks=[]



    for file in get_files(chase):


        name=file.name.lower()



        if (

            "1stho" in name

            or

            "2ndho" in name

        ):

            masks.append(file)

        else:

            images.append(file)




    pairs=pair_images_masks(

        images,

        masks

    )



    print(

        "CHASE pairs:",

        len(pairs)

    )


    return copy_pairs(

        pairs,

        "CHASE_DB1"

    )







# ============================================================
# HRF
# ============================================================


def prepare_hrf():


    print("\nPreparing HRF/HFR")


    hrf = RAW_DATA_DIR / "HFR"


    if not hrf.exists():

        hrf = RAW_DATA_DIR / "HRF"



    if not hrf.exists():

        print("HRF not found")

        return 0



    images=[]

    masks=[]



    for folder in hrf.iterdir():


        if folder.is_dir():


            folder_name = folder.name.lower()


            files=get_files(folder)



            # Images folder

            if (

                folder_name == "images"

                or

                "image" in folder_name

            ):

                images.extend(files)



            # ONLY vessel annotations

            elif (

                "manual" in folder_name

            ):

                masks.extend(files)



    print(

        "HRF images:",

        len(images)

    )


    print(

        "HRF vessel masks:",

        len(masks)

    )



    pairs = pair_images_masks(

        images,

        masks

    )


    print(

        "HRF vessel pairs:",

        len(pairs)

    )



    return copy_pairs(

        pairs,

        "HRF"

    )




# ============================================================
# REPORT
# ============================================================


def dataset_report():


    print("\n")

    print("="*60)

    print(

        "MV-TransUNet Dataset Report"

    )

    print("="*60)



    for name in [

        "DRIVE",

        "STARE",

        "CHASE_DB1",

        "HRF"

    ]:


        images=get_files(

            OUTPUT_DIR/name/"images"

        )


        masks=get_files(

            OUTPUT_DIR/name/"masks"

        )


        print("\n",name)

        print(

            "Images:",

            len(images)

        )


        print(

            "Masks:",

            len(masks)

        )



    print("="*60)







# ============================================================
# MAIN
# ============================================================


def main():


    create_folder(

        OUTPUT_DIR

    )


    prepare_drive()

    prepare_stare()

    prepare_chase()

    prepare_hrf()



    dataset_report()



    print(

        "\nDataset preparation completed."

    )





if __name__=="__main__":

    main()