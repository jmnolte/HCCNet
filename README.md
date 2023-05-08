# Early Stage Hepatocellular Carcinoma Diagnosis Using a 3D Convolutional Neural Network Ensemble

This repository contains the [scripts](https://github.com/jmnolte/thesis/tree/master/scripts) to reproduce all results presented in the project's final [report](https://github.com/jmnolte/thesis/tree/master/report). The report was created as part of the author's master thesis at Utrecht University and was produced in cooperation with Medisch Spectrum Twente and the University of Twente.

## Abstract

For hepatocellular carcinoma (HCC) patients, early-stage diagnosis continues to be the most important predictor of survival. However, lesion detection typically only occurs at an advanced stage and continues to be largely affected by radiologists' experience. To facilitate patient survival, this study proposed a novel computer-aided diagnosis (CAD) system using data on 243 patients (759 observations) with compensated liver cirrhosis. Particularly, four individual 3-dimensional convolutional neural networks were trained, and their predictions aggregated in an ensemble to further enhance accurate tumor diagnosis. The models were validated on an independent internal test set, and their performance was evaluated with respect to a number of standard evaluation metrics. Despite the promising findings attained in our preliminary investigation, the results showed that all four individual models displayed limited diagnostic capabilities, with their predictions remaining largely inferior to the naive prediction of the study's majority class. Meanwhile, the ensemble of the four models displayed greater discriminatory power between HCC and common liver cirrhosis but still failed to accurately infer cancer prevalence. While the proposed CAD thus yielded limited diagnostic value of HCC, its accurate evaluation may have been severely constrained by the study's limitations in computing resources, thus highlighting the need for future research to reevaluate the CAD’s viability.

## Reproducing the Results

### Requirements

To reproduce the results presented in the project's [report](https://github.com/jmnolte/thesis/tree/master/report), access to a Nvidia GPU with a minimum of 8GB VRAM is required. However, training on a single GPU significantly increases processing time. For a faster process, the authors thus recommend the use of four GPUs with 12GB VRAM each. All required package versions are listed in [requirements.txt](https://github.com/jmnolte/thesis/blob/master/requirements.txt). To ensure a fully reproducible workflow, install them by running `pip install -r requirements.txt` in your command line.

### Workflow

All scripts are self contained and should be executed from the command line or using a slurm scheduler in the case of access to a computing cluster. To reproduce the results presented in this study, please execute the scripts in the following order:

1. Model training:
   3D-ResNet10: `torchrun --standalone --nproc_per_node=4 training.py --version resnet10 --pretrained --weighted-sampler --epochs 10 --batch-size 4 --accum-steps 4 --learning-rate 1e-3 --weight-decay 1e-5 --data-dir data_path --results-dir results_path --weights-dir weights_path`
   3D-ResNet18: `torchrun --standalone --nproc_per_node=4 training.py --version resnet18 --pretrained --weighted-sampler --epochs 10 --batch-size 4 --accum-steps 8 --learning-rate 1e-2 --weight-decay 1e-5 --data-dir data_path --results-dir results_path --weights-dir weights_path`
   3D-ResNet34: `torchrun --standalone --nproc_per_node=4 training.py --version resnet34 --pretrained --weighted-sampler --epochs 10 --batch-size 4 --accum-steps 8 --learning-rate 1e-2 --weight-decay 1e-5 --data-dir data_path --results-dir results_path --weights-dir weights_path`
   3D-ResNet50: `torchrun --standalone --nproc_per_node=4 training.py --version resnet10 --pretrained --weighted-sampler --epochs 10 --batch-size 2 --accum-steps 8 --learning-rate 1e-4 --weight-decay 1e-5 --data-dir data_path --results-dir results_path --weights-dir weights_path`
   Ensemble: `torchrun --standalone --nproc_per_node=4 training.py --version ensemble --pretrained --weighted-sampler --epochs 10 --batch-size 2 --accum-steps 4 --learning-rate 1e-2 --data-dir data_path --results-dir results_path --weights-dir weights_path`

## License

The repository is licensed under the `apache-2.0` license.
