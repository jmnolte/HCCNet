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
      <td><a href="https://drive.google.com/file/d/1nz3L3FJRdo8fB3AHVqzO5nkyf9FQFvpg/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-P</td>
      <td>22.0M</td>
      <td>0.309</td>
      <td>0.715</td>
      <td>0.744</td>
      <td>0.936</td>
      <td><a href="https://drive.google.com/file/d/1F9zDDAfxMfzVKSHg8eavjK5DV8t9gGRF/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-N</td>
      <td>45.9M</td>
      <td>0.269</td>
      <td>0.712</td>
      <td>0.689</td>
      <td>0.930</td>
      <td><a href="https://drive.google.com/file/d/17PFeHEHT-yb7nDKY5hcMmeulptSON9bQ/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-T</td>
      <td>72.4M</td>
      <td>0.311</td>
      <td>0.717</td>
      <td>0.624</td>
      <td>0.928</td>
      <td><a href="https://drive.google.com/file/d/1XlgSBT_2-fFyVgSimmjvw8jB9A5yp57S/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td>T1 DCE-MRI</td>
      <td>HCCNet-F</td>
      <td>12.4M</td>
      <td>0.389</td>
      <td>0.755</td>
      <td>0.436</td>
      <td>0.790</td>
      <td><a href="https://drive.google.com/file/d/1UNWm6OhhPFc_e_STdRRzTIPC_pLWtBBy/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-P</td>
      <td>22.0M</td>
      <td>0.402</td>
      <td>0.773</td>
      <td>0.450</td>
      <td>0.777</td>
      <td><a href="https://drive.google.com/file/d/1D2kB7LjmetJjLGAIUlK6pO-kLOJEEHxr/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-N</td>
      <td>45.9M</td>
      <td>0.340</td>
      <td>0.746</td>
      <td>0.388</td>
      <td>0.727</td>
      <td><a href="https://drive.google.com/file/d/1prpkZ6N2wyJZ5_HbCEuSmmSBax3PYVhW/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-T</td>
      <td>72.4M</td>
      <td>0.361</td>
      <td>0.751</td>
      <td>0.535</td>
      <td>0.779</td>
      <td><a href="https://drive.google.com/file/d/1cRX8bJBYVxura9_txlwTWeJtUcdQJJay/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td>T1 IOP & T2-MRI</td>
      <td>HCCNet-F</td>
      <td>12.4M</td>
      <td>0.427</td>
      <td>0.672</td>
      <td>0.223</td>
      <td>0.615</td>
      <td><a href="https://drive.google.com/file/d/1wkoDKHlBxTyYyuy2TBG6XWQyOu8hjZDc/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-P</td>
      <td>22.0M</td>
      <td>0.349</td>
      <td>0.675</td>
      <td>0.282</td>
      <td>0.666</td>
      <td><a href="https://drive.google.com/file/d/1XcYeggfQKC7RE3ZAKGZ7E8G97djEtLcX/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-N</td>
      <td>45.9M</td>
      <td>0.319</td>
      <td>0.642</td>
      <td>0.338</td>
      <td>0.703</td>
      <td><a href="https://drive.google.com/file/d/1VYXmZzFYNfCtd160JBtjXjSA_pphwWCV/view?usp=sharing" >model weights</a></td>
    </tr>
    <tr>
      <td></td>
      <td>HCCNet-T</td>
      <td>72.4M</td>
      <td>0.298</td>
      <td>0.632</td>
      <td>0.268</td>
      <td>0.665</td>
      <td><a href="https://drive.google.com/file/d/1y-kkamZEHqGJbhjzUOaCpzbuXibJZQN-/view?usp=sharing" >model weights</a></td>
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
