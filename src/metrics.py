"""
MV-TransUNet Evaluation Metrics

Metrics:

- Dice Score
- Sensitivity
- Specificity
- Accuracy
- ROC-AUC
- Thin Vessel Dice

Designed for:
Retinal Vessel Segmentation


Author:
MV-TransUNet Research Project
"""


import numpy as np

import cv2

from sklearn.metrics import roc_auc_score

from skimage.morphology import skeletonize





# ============================================================
# BASIC CONFUSION MATRIX
# ============================================================


def confusion_matrix(prediction, target):


    prediction = prediction.astype(bool)

    target = target.astype(bool)



    TP = np.logical_and(

        prediction,

        target

    ).sum()



    TN = np.logical_and(

        np.logical_not(prediction),

        np.logical_not(target)

    ).sum()



    FP = np.logical_and(

        prediction,

        np.logical_not(target)

    ).sum()



    FN = np.logical_and(

        np.logical_not(prediction),

        target

    ).sum()



    return TP,TN,FP,FN







# ============================================================
# DICE
# ============================================================


def dice_score(prediction,target):


    TP,TN,FP,FN = confusion_matrix(

        prediction,

        target

    )


    dice = (

        2*TP

        /

        (

            2*TP

            +

            FP

            +

            FN

            +

            1e-8

        )

    )


    return dice







# ============================================================
# SENSITIVITY
# ============================================================


def sensitivity(prediction,target):


    TP,TN,FP,FN = confusion_matrix(

        prediction,

        target

    )


    return (

        TP

        /

        (

            TP

            +

            FN

            +

            1e-8

        )

    )








# ============================================================
# SPECIFICITY
# ============================================================


def specificity(prediction,target):


    TP,TN,FP,FN = confusion_matrix(

        prediction,

        target

    )


    return (

        TN

        /

        (

            TN

            +

            FP

            +

            1e-8

        )

    )







# ============================================================
# ACCURACY
# ============================================================


def accuracy(prediction,target):


    TP,TN,FP,FN = confusion_matrix(

        prediction,

        target

    )


    return (

        TP+TN

        /

        (

            TP

            +

            TN

            +

            FP

            +

            FN

            +

            1e-8

        )

    )







# ============================================================
# AUC
# ============================================================


def auc_score(probability,target):


    probability = probability.flatten()

    target = target.flatten()



    return roc_auc_score(

        target,

        probability

    )







# ============================================================
# THIN VESSEL EXTRACTION
# ============================================================


def extract_thin_vessels(

        mask,

        width_threshold=3

):


    """
    Extract vessels <=3 pixels width

    Used for micro-vessel evaluation.
    """



    mask = mask.astype(

        np.uint8

    )



    skeleton = skeletonize(

        mask

    )



    distance = cv2.distanceTransform(

        mask,

        cv2.DIST_L2,

        5

    )



    vessel_width = distance*2



    thin_region = (

        vessel_width <= width_threshold

    )



    thin_vessels = np.logical_and(

        skeleton,

        thin_region

    )



    return thin_vessels







# ============================================================
# THIN VESSEL DICE
# ============================================================


def thin_vessel_dice(

        prediction,

        target

):


    target_thin = extract_thin_vessels(

        target

    )



    prediction_thin = np.logical_and(

        prediction,

        target_thin

    )



    return dice_score(

        prediction_thin,

        target_thin

    )








# ============================================================
# COMPLETE METRIC REPORT
# ============================================================


def calculate_metrics(

        prediction,

        target,

        probability=None

):


    results = {}



    results["dice"] = dice_score(

        prediction,

        target

    )



    results["sensitivity"] = sensitivity(

        prediction,

        target

    )



    results["specificity"] = specificity(

        prediction,

        target

    )



    results["accuracy"] = accuracy(

        prediction,

        target

    )



    results["thin_vessel_dice"] = thin_vessel_dice(

        prediction,

        target

    )



    if probability is not None:


        results["auc"] = auc_score(

            probability,

            target

        )


    return results







# ============================================================
# TEST
# ============================================================


if __name__ == "__main__":


    print(
        "Testing MV-TransUNet Metrics"
    )


    prediction=np.random.randint(

        0,

        2,

        (

            256,

            256

        )

    )


    target=np.random.randint(

        0,

        2,

        (

            256,

            256

        )

    )


    probability=np.random.random(

        (

            256,

            256

        )

    )



    results = calculate_metrics(

        prediction,

        target,

        probability

    )



    print()



    for key,value in results.items():

        print(

            key,

            ":",

            round(value,4)

        )