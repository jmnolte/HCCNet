# Detecting hepatocellular carcinoma in non-alcoholic fatty liver disease patients using deep learning based radiomics

This repository contains the data along with the corresponding python code to reproduce all results presented in the project's preliminary [research report](https://github.com/jmnolte/thesis/tree/master/report). The research report was created as part of the author's master thesis at the University of Utrecht.

## Abstract

Early stage diagnosis could reduce the burden of hepatocellular carcimonia (HCC). The disease is the most common liver cancer in adults, but remains difficult to diagnose. Coincidently, convolutional neural networks have demonstrated state-of-the-art performance in many fields of medical imaging, but have found sparse employment in HCC diagnosis. Therefore, this study adopts a series of pretrained convolutional neural architectures and retrains them on a cohort of 40 non-alcoholic liver disease patients. The performance of the deep learning models is compared to a baseline radiomics model, which is derived using handcrafted radiomic features. Additionally, the interpretation of the deep learning models is augmented, modeling a subset of deep features that can be interpreted in the context of handcrafted radiomics. Although, as of writing the report, no hyperparameter optimization has been performed, the preliminary results show that the deep learning models achieve a slightly higher classification performance than the baselline radiomic model. Furthermore, the results indicate that the loss of information induced by solely modeling the features that can be interpreted in the context of handcrafted radiomics is not reflected in the model's performance metrics.

## License

The repository is licensed under the `apache-2.0` license.
