# HCCNet: Early Prediction of Hepatocellular Carcinoma Using Longitudinal MRI

## Abstract

For hepatocellular carcinoma (HCC) patients, early-stage diagnosis continues to be the most important predictor of survival. However, lesion detection typically only occurs at an advanced stage and is largely affected by radiologists' experience. Therefore, this repository implements HCCNet - a spatio-temporal neural network that utilizes a 3D ConvNeXt [^1] backbone combined with a transformer encoder [^2] - to predict future cancer development based on past MRI examinations.

## Model Architecture

![Model Architecture](https://github.com/jmnolte/thesis/blob/master/report/architecture.png)

The model embeds the raw MR images in higher-dimensional feature space, adds time-based positional encodings, and subsequently processes the sequence of embedded image representations to predict the likelihood of future HCC development.

## Training Protocol

Model training follows a step-wise approach. That is, we first pretrain both CNN backbone and Transformer encoder using self-supervised learning and then fine-tune the full model on our downstream task.

### Pre-Training

We pretrain the CNN backbone adapting the DINO pretraining framework proposed in [^3] to 3D medical images. Furthermore, for the Transformer encoder, we employ a custom pre-training approach, where we randomly shuffle 50% of embedded image sequences and train the model to differeniate shuffled from non-shuffled sequences. Note that during Transformer pre-training, we keep the parameters of the CNN backbone froozen.

### Fine-Tuning

After pre-training, we initialize the model with the weights obtained after pre-training, add a linear pooling layer to the architecture, and fine-tune the full model on our downstream task.

## Results

| Modality | Init weights | AUC-PR | AUC-ROC |
| --- | --- | --- | --- | 
| Diffusion MRI | Random | 0.426 | 0.776 |
| T1 pre- and post contrast MRI | Random | 0.360 | 0.768 |
| T1 in- and out-of-phase MRI | Random | 0.202 | 0.621 |
| T2 MRI | Random | 0.374 | 0.705 |

Models results are averaged over 10 runs with different random seeds.

## Future Additions

- [x] Add baseline results, i.e., without pre-training, for diffusion, T1, and T2 weighted MRI
- [ ] Add results after pre-training

## License

The repository is licensed under the `apache-2.0` license.

[^1]: Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S., 2022. A ConvNet for the 2020s. URL: http://arxiv.org/abs/2201.03545. arXiv:2201.03545.
[^2]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I., 2023. Attention Is All You Need. URL: http://arxiv.org/abs/1706.03762. arXiv:1706.03762.
[^3]: Caron, M., Touvron, H., Misra, I., J ́egou, H., Mairal, J., Bojanowski, P., Joulin, A., 2021. Emerging Properties in Self-Supervised Vision Transformers. URL: http://arxiv.org/abs/2104.14294. arXiv:2104.14294.
