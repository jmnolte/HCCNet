# HCCNet: Early Prediction of Hepatocellular Carcinoma Using Longitudinal MRI

## Abstract

For hepatocellular carcinoma (HCC) patients, early-stage diagnosis continues to be the most important predictor of survival. However, lesion detection typically only occurs at an advanced stage and is largely affected by radiologists' experience. Therefore, this repository implements HCCNet - a spatio-temporal neural network that utilizes a 3D ConvNeXt backbone combined with a custom transformer encoder - to predict future cancer development based on past MRI examinations.

## Model Architecture

![Model Architecture](https://github.com/jmnolte/thesis/blob/master/report/architecture.png)

The model embeds the raw MR images in higher-dimensional feature space, adds time-based positional encodings, and subsequently processes the sequence of embedded image representations to predict the likelihood of future HCC development.

## Training Protocol

Model training follows a step-wise approach. That is, we first pretrain both CNN backbone and Transformer encoder using self-supervised learning and then fine-tune the full model on our downstream task.

### Pre-Training

We pretrain the CNN backbone adapting the DINO pretraining framework proposed in [^1] to 3D medical images. Furthermore, for the Transformer encoder, we employ a custom variant of the original masked autoencoder pre-training approach [^2]. As such, we mask 60% of the image embeddings obtained from the pretrained CNN backbone and recover their original feature representation using a simple one-layer decoder.

### Fine-Tuning

After pre-training, we initialize the model with the weights obtained after pre-training, add a linear pooling layer to the architecture, and fine-tune the full model on our downstream task.

## Future Additions

- [x] Add baseline results, i.e., without pre-training, for diffusion weighted MRI
- [ ] Add baseline results for contrast-enhanced T1, T2, and in- and out-of-phase T1 weighted MRIs
- [ ] Add results after pre-training

## License

The repository is licensed under the `apache-2.0` license.

[^1]: Caron, M., Touvron, H., Misra, I., J ́egou, H., Mairal, J., Bojanowski, P., Joulin, A., 2021. Emerging Properties in Self-Supervised Vision Transformers. URL: http://arxiv.org/abs/2104.14294. arXiv:2104.14294.
[^2]: Kaiming, H., Xinlei, C., Xie, S., Li, Y., Doll, P., Girshick, R., 2021. Masked Autoencoders Are Scalable Vision Learners. URL: http://arxiv.org/abs/2111.06377. arXiv:2111.06377.
