"""
MV-TransUNet Dataset Preparation Pipeline

Supported:
- DRIVE
- STARE
- CHASE_DB1
- HRF/HFR


Output:

datasets_processed/

    DRIVE_train/
        images/
        masks/

    DRIVE_test/
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


IMPORTANT -- DRIVE_train / DRIVE_test separation
-------------------------------------------------
An earlier version of this script merged DRIVE/training and DRIVE/test
into a single combined "DRIVE" output folder before any train/validation
split occurred downstream. This silently destroyed the standard DRIVE
evaluation protocol: any 80/20 split subsequently drawn from that merged
pool could -- and very likely did -- train on images every published
DRIVE baseline reserves exclusively for held-out testing, making any
reported Dice score non-comparable to the literature and, more
seriously, contaminated by test-set leakage.

This version keeps DRIVE_train and DRIVE_test as two entirely separate
output folders, exactly mirroring the official DRIVE directory
structure. `dataset.train_dataset` in config.yaml should point at
DRIVE_train (used for the 80/20 train/validation split during training),
and a new `dataset.test_dataset` should point at DRIVE_test (used ONLY
by evaluate.py, never touched during training or model selection).
"""


from pathlib import Path

import shutil

import cv2

import numpy as np



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



def check_mask_looks_like_vessels(mask_path, split_name, max_plausible_fraction=0.30):
    """
    Sanity check against exactly the failure mode this rewrite fixes:
    DRIVE (and several other retinal datasets) ship TWO different
    things that both get casually called "mask" -- the field-of-view
    (FOV) circle, which is ~70-90% foreground, and the actual vessel
    annotation, which is typically 5-15% foreground. Silently pairing
    images with the FOV circle instead of the vessel annotation
    produces a dataset that trains and validates without error, but
    is not a vessel segmentation dataset at all.

    This loads one prepared mask and checks its foreground fraction.
    A high fraction almost certainly means the wrong source folder
    was used. This does not replace looking at the images yourself --
    it is a cheap automatic backstop, not a substitute for the manual
    verification that caught this bug in the first place.
    """

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"WARNING: could not read {mask_path} for sanity check.")
        return

    foreground_fraction = float((mask > 127).mean())

    print(
        f"{split_name} mask sanity check ({mask_path.name}): "
        f"foreground fraction = {foreground_fraction:.3f}"
    )

    if foreground_fraction > max_plausible_fraction:
        print(
            f"WARNING: {split_name} foreground fraction "
            f"({foreground_fraction:.3f}) is implausibly high for a "
            f"vessel annotation (expected roughly 0.05-0.15). This "
            f"looks like a field-of-view (FOV) mask, not a vessel "
            f"mask. Check masks_subfolder for this split -- do not "
            f"proceed to training until this is resolved."
        )




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
# DRIVE -- TRAIN AND TEST KEPT SEPARATE
# ============================================================


def prepare_drive_split(

    source_subfolder,

    output_name,

    images_subfolder,

    masks_subfolder,

    expected_count=20,

):
    """
    Prepare exactly ONE DRIVE split (training OR test) into its own
    dedicated output folder. Never call this for both splits into the
    same output_name -- that is precisely the merging bug an earlier
    version of this script had.

    images_subfolder / masks_subfolder are EXPLICIT, per-split
    relative paths (may include a subdirectory, e.g.
    "manual/1st_manual") rather than assumed constants. DRIVE raw
    downloads/re-organizations are not guaranteed to use the same
    folder names or nesting for training vs. test -- verify these by
    listing the actual raw folders before trusting any default here.

    Also verify masks_subfolder points at the VESSEL annotation, not
    the field-of-view (FOV) circle -- both are commonly just called
    "mask(s)" in different DRIVE redistributions, and pairing images
    with the FOV circle instead of the vessel annotation produces a
    dataset that trains without any error while being completely
    wrong. See check_mask_looks_like_vessels below, which is run
    automatically on every prepared split as a backstop -- but the
    authoritative check is looking at the images yourself.
    """

    print(

        f"\nPreparing DRIVE ({source_subfolder} -> {output_name})"

    )



    folder = RAW_DATA_DIR / "DRIVE" / source_subfolder



    images = get_files(

        folder / images_subfolder

    )



    masks = get_files(

        folder / masks_subfolder

    )



    print(

        f"{output_name} images:",

        len(images),

        f"(from {images_subfolder})",

    )


    print(

        f"{output_name} masks:",

        len(masks),

        f"(from {masks_subfolder})",

    )



    if expected_count is not None and len(images) != expected_count:

        print(

            f"WARNING: expected {expected_count} images in "

            f"DRIVE/{source_subfolder}/{images_subfolder}, found "

            f"{len(images)}. Verify raw dataset integrity before "

            "trusting any downstream train/validation split or "

            "reported metric."

        )



    pairs = pair_images_masks(

        images,

        masks

    )



    print(

        f"{output_name} pairs:",

        len(pairs)

    )



    count = copy_pairs(

        pairs,

        output_name

    )



    if count > 0:

        first_mask_path = (

            OUTPUT_DIR / output_name / "masks"

        )

        first_mask_files = get_files(first_mask_path)

        if first_mask_files:

            check_mask_looks_like_vessels(

                first_mask_files[0],

                output_name,

            )



    return count





def prepare_drive_train():

    return prepare_drive_split(

        source_subfolder="training",

        output_name="DRIVE_train",

        # Verified against the actual raw folder listing:
        # training/ uses "image" (singular) for images, and the
        # vessel annotation is directly at training/1st_manual/ (flat,
        # same nesting level as test/1st_manual/) -- NOT
        # training/masks, which is the FOV circle, and NOT nested
        # under a "manual" folder as an earlier version of this
        # function incorrectly assumed.
        images_subfolder="image",

        masks_subfolder="1st_manual",

    )





def prepare_drive_test():

    return prepare_drive_split(

        source_subfolder="test",

        output_name="DRIVE_test",

        # Verified against the actual raw folder listing:
        # test/ uses "images" (plural), and the vessel annotation is
        # directly at test/1st_manual/ (no "manual" nesting level,
        # unlike training/) -- NOT test/masks, which is the FOV
        # circle.
        images_subfolder="images",

        masks_subfolder="1st_manual",

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

        "DRIVE_train",

        "DRIVE_test",

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



    print(

        "\nDRIVE_train and DRIVE_test are now separate folders. "

        "Update config.yaml: dataset.train_dataset -> DRIVE_train, "

        "dataset.test_dataset -> DRIVE_test. Any model previously "

        "trained against the old merged datasets_processed/DRIVE/ "

        "folder must be retrained -- its validation/early-stopping "

        "checkpoint selection may have used images now assigned to "

        "DRIVE_test."

    )







# ============================================================
# MAIN
# ============================================================


def main():


    create_folder(

        OUTPUT_DIR

    )


    prepare_drive_train()

    prepare_drive_test()

    prepare_stare()

    prepare_chase()

    prepare_hrf()



    dataset_report()



    print(

        "\nDataset preparation completed."

    )





if __name__=="__main__":

    main()