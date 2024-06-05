# HCCNet: Early Prediction of Hepatocellular Carcinoma Using Longitudinal MRI

## Abstract

For hepatocellular carcinoma (HCC) patients, early-stage diagnosis continues to be the most important predictor of survival. However, lesion detection typically only occurs at an advanced stage and is largely affected by radiologists' experience. Therefore, this repository implements HCCNet - a spatio-temporal neural network that utilizes a 3D ConvNeXt [^1] backbone combined with a transformer encoder [^2] - to predict future cancer development based on past MRI examinations.

## Model Architecture

![Model Architecture](https://github.com/jmnolte/thesis/blob/master/report/architecture.png)

The model embeds the raw MR images in higher-dimensional feature space, adds time-based positional encodings, and subsequently processes the sequence of embedded image representations to predict the likelihood of future HCC development.

## Training Protocol

Model training follows a step-wise approach. That is, we first pretrain both CNN backbone and Transformer encoder using self-supervised learning and then fine-tune the full model on our downstream task.

### Pre-Training

We pretrain the CNN backbone adapting the DINO pretraining framework proposed in [^3] to 3D medical images. Furthermore, for the Transformer encoder, we employ a custom pre-training approach, where we randomly shuffle 50% of embedded image sequences and train the model to differentiate shuffled from non-shuffled sequences. Note that during Transformer pre-training, we keep the parameters of the CNN backbone frozen.

### Fine-Tuning

After pre-training, we initialize the model with the weights obtained after pre-training, add a linear pooling layer to the architecture, and fine-tune the full model on our downstream task. We train the model over 10 runs with different random seeds and ensemble the runs' predictions by averaging over the predicted probabilities. 

## Results

<table>
  <thead>
    <tr>
      <th colspan="3"></th>
      <th colspan="2">Random init</th>
      <th colspan="2">Pretrained init</th>
      <th colspan="1"></th>
    </tr>
    <tr>
      <th>Modality</th>
      <th>Architecture</th>
      <th>Params</th>
      <th>AUPRC</th>
      <th>AUROC</th>
      <th>AUPRC</th>
      <th>AUROC</th>
      <th>Downloads</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>DW-MRI</td>
      <td>HCCNet-F</td>
      <td>12.4M</td>
      <td>0.286</td>
      <td>0.706</td>
      <td>0.690</td>
      <td>0.916</td>
      <td><a href="https://drive.google.com/file/d/1NE32AjbCksuk_0DMJgyRGmQjP0loNYfw/view?usp=share_link" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-P</td>
      <td>22.0M</td>
      <td>0.309</td>
      <td>0.715</td>
      <td>0.744</td>
      <td>0.936</td>
      <td><a href="https://drive.google.com/file/d/1R8AiBwWq3VMRwgU32AWCFqL8luNckV81/view?usp=share_link" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-N</td>
      <td>45.9M</td>
      <td>0.269</td>
      <td>0.712</td>
      <td>0.689</td>
      <td>0.930</td>
      <td><a href="https://drive.google.com/file/d/1PRxlPqMqF17RsQJ9lW6tAsxcG7kX8bDI/view?usp=share_link" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-T</td>
      <td>72.4M</td>
      <td>0.311</td>
      <td>0.717</td>
      <td>0.624</td>
      <td>0.928</td>
      <td><a href="https://drive.google.com/file/d/1TnFdq85wVLL_T17jo3pOSvvBjtua9KAJ/view?usp=share_link" >model weights</a></td>
    </tr>
    <tr>
      <td>T1 DCE-MRI</td>
      <td>HCCNet-F</td>
      <td>12.4M</td>
      <td>0.356</td>
      <td>0.740</td>
      <td>0.442</td>
      <td>0.792</td>
      <td><a href="https://drive.google.com/file/d/1VIIfsGj2NwPkB5NMrPpPmDQ5VymUnFA-/view?usp=share_link" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-P</td>
      <td>22.0M</td>
      <td>0.371</td>
      <td>0.753</td>
      <td>0.478</td>
      <td>0.801</td>
      <td><a href="https://drive.google.com/file/d/10HtFZ8miAN7z3U2V7R-cs5CsBGlbeUje/view?usp=share_link" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-N</td>
      <td>45.9M</td>
      <td>0.366</td>
      <td>0.746</td>
      <td>0.463</td>
      <td>0.764</td>
      <td><a href="https://drive.google.com/file/d/1Ic952xETjs-dyHD5Sw18BT4hPBdQIHJS/view?usp=share_link" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-T</td>
      <td>72.4M</td>
      <td>0.385</td>
      <td>0.769</td>
      <td>0.521</td>
      <td>0.827</td>
      <td><a href="https://drive.google.com/file/d/1wXpCS8sdYbQ18jdB6hVKIm6fKZSeuaq_/view?usp=share_link" >model weights</a></td>
    </tr>
  </tbody>
</table>

Diffusion weighted MRI's (i.e., DW-MRI) compose a four channel image with diffusion coefficients `b = 0, 150, 400, and 800`. Contrarily, dynamic contrast enhanced T1 weighted MRIs comprise of one pre-contrast and three post-contrast (i.e., late aterial, portal venous, and delayed phase) scans.

## Future Additions

- [x] Add results for diffusion MRI
- [x] Add results for T1 contrast enhanced MRI
- [ ] Add results for T1 in- and out-of-phase MRI
- [ ] Add results for T2 MRI

## License

The repository is licensed under the `apache-2.0` license.

[^1]: Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S., 2022. A ConvNet for the 2020s. URL: http://arxiv.org/abs/2201.03545. arXiv:2201.03545.
[^2]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I., 2023. Attention Is All You Need. URL: http://arxiv.org/abs/1706.03762. arXiv:1706.03762.
[^3]: Caron, M., Touvron, H., Misra, I., J ́egou, H., Mairal, J., Bojanowski, P., Joulin, A., 2021. Emerging Properties in Self-Supervised Vision Transformers. URL: http://arxiv.org/abs/2104.14294. arXiv:2104.14294.
