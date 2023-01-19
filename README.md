# Detecting hepatocellular carcinoma in non-alcoholic fatty liver disease patients using deep learning based radiomics

This repository contains the [data](https://github.com/jmnolte/thesis/tree/master/test_data) along with the corresponding [code](https://github.com/jmnolte/thesis/tree/master/test_scripts) to reproduce all results presented in the project's preliminary [research report](https://github.com/jmnolte/thesis/tree/master/report). The research report was created as part of the author's master thesis at the University of Utrecht.

## Abstract

Early stage diagnosis could reduce the burden of hepatocellular carcimonia (HCC). The disease is the most common liver cancer in adults, but remains difficult to diagnose. Coincidently, convolutional neural networks have demonstrated state-of-the-art performance in many fields of medical imaging, but have found sparse employment in HCC diagnosis. Therefore, this study adopts a series of pretrained convolutional neural architectures and retrains them on a cohort of 40 non-alcoholic liver disease patients. The performance of the deep learning models is compared to a baseline radiomics model, which is derived using handcrafted radiomic features. Additionally, the interpretation of the deep learning models is augmented, modeling a subset of deep features that can be interpreted in the context of handcrafted radiomics. Although, as of writing the report, no hyperparameter optimization has been performed, the preliminary results show that the deep learning models achieve a slightly higher classification performance than the baselline radiomic model. Furthermore, the results indicate that the loss of information induced by solely modeling the features that can be interpreted in the context of handcrafted radiomics is not reflected in the model's performance metrics.

## Reproducing the Results

### Requirements

To reproduce the results presented in the project's [research report](https://github.com/jmnolte/thesis/tree/master/report) a Google Drive account as well as access to Google Colab are required. All required package versions are listed in [requirements.txt](https://github.com/jmnolte/thesis/blob/master/requirements.txt). To ensure a fully reproducible workflow, install them by running `pip install -r requirements.txt` in your command line.

### Workflow

The scripts for the [baseline](https://github.com/jmnolte/thesis/tree/master/test_scripts/baseline_approach) and the [deep learning](https://github.com/jmnolte/thesis/tree/master/test_scripts/deep_learning_approach) approach are self contained and can be executed independently. The scripts for the baseline approach are executed in the following order:

1. [segmentation.ipynb](https://github.com/jmnolte/thesis/blob/master/test_scripts/baseline_approach/segmentation.ipynb)
2. [segmentation_visual.ipynb](https://github.com/jmnolte/thesis/blob/master/test_scripts/baseline_approach/segmentation_visual.ipynb) (optional)
3. [feature_extraction.ipynb](https://github.com/jmnolte/thesis/blob/master/test_scripts/baseline_approach/feature_extraction.ipynb)
4. [model_building.ipynb](https://github.com/jmnolte/thesis/blob/master/test_scripts/baseline_approach/model_building.ipynb)

## License

The repository is licensed under the `apache-2.0` license.
