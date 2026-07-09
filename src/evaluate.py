"""
MV-TransUNet Evaluation Pipeline

Retinal Vessel Segmentation Framework

Metrics:
    - Dice
    - Sensitivity
    - Specificity
    - Accuracy
    - ROC-AUC

Additional:
    - Thin vessel Dice
    - Skeletonization analysis
    - Cross dataset evaluation

Supported:
    - DRIVE
    - STARE
    - CHASE_DB1
    - HRF

"""


import os

import yaml

import numpy as np


from tqdm import tqdm


import torch

import cv2


from sklearn.metrics import roc_auc_score


from skimage.morphology import skeletonize


from models.mv_transunet import MVTransUNet


from src.datasets import build_test_loader





# ============================================================
# CONFIGURATION
# ============================================================


def load_config(path):


    with open(path,"r") as file:


        return yaml.safe_load(file)





# ============================================================
# METRICS
# ============================================================


def calculate_metrics(

    prediction,

    target

):


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



    dice = (

        2 * TP

        /

        (

            2 * TP

            +

            FP

            +

            FN

            +

            1e-8

        )

    )



    sensitivity = (

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



    specificity = (

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



    accuracy = (

        TP + TN

    ) / (

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



    return {


        "dice": dice,


        "sensitivity": sensitivity,


        "specificity": specificity,


        "accuracy": accuracy


    }





# ============================================================
# THIN VESSEL ANALYSIS
# ============================================================


def extract_thin_vessels(

    mask,

    threshold=3

):


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



    vessel_width = distance * 2



    thin_region = (

        vessel_width <= threshold

    )



    thin_vessels = np.logical_and(

        skeleton,

        thin_region

    )



    return thin_vessels





def calculate_thin_vessel_dice(

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


    metric = calculate_metrics(

        prediction_thin,

        target_thin

    )


    return metric["dice"]





# ============================================================
# MODEL LOADING
# ============================================================


def load_model(

    checkpoint_path,

    device

):


    model = MVTransUNet(

        pretrained=False

    )



    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )



    model.load_state_dict(

        checkpoint["model_state"]

    )



    model.to(device)



    model.eval()



    return model





# ============================================================
# EVALUATION FUNCTION
# ============================================================


def evaluate_dataset(

    model,

    loader,

    device,

    save_directory=None

):


    results=[]



    all_probabilities=[]


    all_targets=[]



    if save_directory:


        os.makedirs(

            save_directory,

            exist_ok=True

        )



    with torch.no_grad():


        for index,batch in enumerate(

            tqdm(

                loader,

                desc="Evaluating"

            )

        ):



            images = batch["image"].to(

                device,

                non_blocking=True

            )



            masks = batch["mask"].numpy()



            with torch.cuda.amp.autocast(

                enabled=device.type=="cuda"

            ):


                outputs = model(

                    images

                )



            probabilities = torch.sigmoid(

                outputs

            ).cpu().numpy()



            predictions = (

                probabilities > 0.5

            )



            for sample_id,(

                pred,

                target,

                probability

            ) in enumerate(zip(

                predictions,

                masks,

                probabilities

            )):



                pred = pred.squeeze()


                target = target.squeeze()


                probability = probability.squeeze()



                metrics = calculate_metrics(

                    pred,

                    target

                )



                metrics["thin_dice"] = calculate_thin_vessel_dice(

                    pred,

                    target

                )



                results.append(

                    metrics

                )



                all_probabilities.extend(

                    probability.flatten()

                )



                all_targets.extend(

                    target.flatten()

                )



                if save_directory:



                    output_path = os.path.join(

                        save_directory,

                        f"prediction_{index}_{sample_id}.png"

                    )



                    cv2.imwrite(

                        output_path,

                        (

                            pred.astype(np.uint8)

                            *

                            255

                        )

                    )





    final_metrics={}



    for key in results[0]:


        final_metrics[key]=float(

            np.mean(

                [

                    item[key]

                    for item in results

                ]

            )

        )



    try:


        final_metrics["auc"]=float(

            roc_auc_score(

                all_targets,

                all_probabilities

            )

        )


    except ValueError:


        final_metrics["auc"]=0.0



    return final_metrics





# ============================================================
# SAVE RESULTS
# ============================================================


def save_results(

    results,

    path

):


    with open(

        path,

        "w"

    ) as file:


        yaml.dump(

            results,

            file,

            sort_keys=False

        )





# ============================================================
# MAIN
# ============================================================


def main():


    config = load_config(

        "config.yaml"

    )



    device=torch.device(

        "cuda"

        if torch.cuda.is_available()

        else

        "cpu"

    )



    print(

        "Evaluation device:",

        device

    )



    model = load_model(

        "./checkpoints/best_model.pth",

        device

    )



    image_size = config["preprocessing"]["image_size"]["height"]



    datasets = config["dataset"]["cross_dataset_testing"]["datasets"]



    all_results={}



    # Evaluate DRIVE validation

    validation_loader = build_test_loader(

        image_dir=config["dataset"]["validation_dataset"]["image_dir"],

        mask_dir=config["dataset"]["validation_dataset"]["mask_dir"],

        image_size=image_size

    )



    print("\nEvaluating DRIVE")



    drive_results=evaluate_dataset(

        model,

        validation_loader,

        device,

        "./experiments/predictions/DRIVE"

    )



    all_results["DRIVE"]=drive_results





    # Cross dataset evaluation

    for dataset in datasets:


        name=dataset["name"]



        print(

            f"\nEvaluating {name}"

        )



        loader=build_test_loader(

            image_dir=dataset["image_dir"],

            mask_dir=dataset["mask_dir"],

            image_size=image_size

        )



        metrics=evaluate_dataset(

            model,

            loader,

            device,

            f"./experiments/predictions/{name}"

        )



        all_results[name]=metrics





    os.makedirs(

        "./experiments",

        exist_ok=True

    )



    save_results(

        all_results,

        "./experiments/evaluation_results.yaml"

    )



    print("\n==============================")

    print("Evaluation Complete")

    print("==============================")



    for dataset,metrics in all_results.items():


        print("\n",dataset)



        for key,value in metrics.items():


            print(

                key,

                ":",

                round(value,4)

            )





if __name__=="__main__":


    main()