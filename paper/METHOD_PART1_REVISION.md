# Part 1 Revision: Reconstruction-Oriented MAE Visual Encoder

## Scope Positioning

For the full paper, it is acceptable to describe the overall model as a **VLM-inspired multimodal forecasting architecture**. However, in **Part 1**, the focus should remain on the **visual branch** rather than the whole multimodal interface. In other words:

- At the architecture level, you may state that the full framework follows a VLM-style design.
- In this subsection, the module should be described as a **reconstruction-oriented MAE visual encoder**, not as a full VLM encoder.

This distinction is important because the core contribution of this part is the MAE-based visual representation learning for time-series-derived images.

## Main Revision Principles

The original paragraph should be revised according to the following principles:

1. Replace `VLM encoder` with `reconstruction-oriented MAE visual encoder` when describing this specific subsection.
2. Do not make `dual-path` the main narrative of Part 1. If needed, mention it only as an optional extension or variant.
3. Remove the statement that the text embedding is a zero placeholder from this subsection. That belongs to the later fusion/interface description.
4. Avoid claiming that MAE is directly optimized end-to-end by the forecasting loss unless the implementation truly updates the MAE backbone in the forecasting stage.
5. Avoid absolute claims such as "highly robust" unless supported by dedicated robustness experiments. Use more cautious scientific phrasing such as `is expected to` or `is designed to`.

## Minimally Revised Text

The following version keeps as much of your original wording as possible while correcting the main issues discussed earlier.

```text
Although the overall framework follows a VLM-inspired multimodal architecture, this part focuses specifically on the visual branch. The reconstruction-oriented MAE visual encoder is responsible for extracting visual features. Compared with conventional visual backbones, the MAE encoder is better suited for handling the abstract pixel-texture images derived from the transformation of time-series data.

We first convert the normalized multivariate time series into an image using a learnable mapper. This provides a consistent visual input for the encoder without requiring external image data. The converted image is then fed into a pretrained MAE encoder. During pretraining, MAE learns to preserve fine-grained structure by reconstructing heavily masked images. This property makes it particularly suitable for time-series-derived images and is expected to improve tolerance to artifacts or noise introduced by the TS-to-Image conversion.

To further exploit this property, we do not rely only on a standard discriminative summary representation. Instead, we construct reconstruction-conditioned visual features from the masked-image recovery process. In this way, the resulting visual embedding becomes more sensitive to temporal structures such as periodicity, trends, and discontinuities. The pooled reconstruction-oriented MAE output is used as the visual embedding $z_v \in \mathbb{R}^{B \times H}$.

During the forecasting stage, the pretrained MAE mainly serves as a reconstruction-oriented visual feature extractor. The forecasting framework is optimized with the prediction loss, while no additional reconstruction loss is jointly imposed as an auxiliary objective in this stage.
```

## Original-to-Revised Mapping

If you prefer to revise sentence by sentence rather than replace the whole paragraph, use the following substitutions.

- Original: `The VLM encoder is responsible for extracting multimodal features.`
  Replace with: `Although the overall framework follows a VLM-inspired multimodal architecture, this part focuses specifically on the visual branch, where the reconstruction-oriented MAE visual encoder is responsible for extracting visual features.`

- Original: `In comparison to traditional visual language models, the MAE encoder is better suited for handling the abstract pixel texture images derived from the transformation of time - series data.`
  Replace with: `Compared with conventional visual backbones, the MAE encoder is better suited for handling the abstract pixel-texture images derived from the transformation of time-series data.`

- Original: `This design makes it highly robust to artifacts or noise that may arise from the TS-to-Image conversion.`
  Replace with: `This property is expected to improve tolerance to artifacts or noise that may arise from the TS-to-Image conversion.`

- Original: `Instead of relying on standard discriminative features like CLS token, we construct reconstruction-oriented features by mixing encoder latent tokens with decoder or intermediate reconstruction representations implemented via a dual-path design.`
  Replace with: `To further exploit this property, we construct reconstruction-conditioned visual features from the masked-image recovery process rather than relying only on a standard discriminative summary representation.`

- Original: `This yields reconstruction-aware embeddings that better align with temporal structure such as periodicity, trends, and discontinuities.`
  Keep, but revise slightly to: `This yields reconstruction-oriented visual embeddings that are more sensitive to temporal structures such as periodicity, trends, and discontinuities.`

- Original: `In this MAE configuration, the text embedding z_t in R^{B x H} is a zero-placeholder to maintain computational graph integrity.`
  Remove from Part 1 and move to the later fusion/interface subsection if still needed.

- Original: `The VLM encoder then passes the visual embedding ... and the placeholder text ... to the fusion module.`
  Remove from Part 1 and move to the later multimodal fusion subsection.

- Original: `MAE's reconstruction objective is used only in pretraining. In the end-to-end forecasting stage, the MAE serves as a visual feature extractor and is optimized primarily by the forecasting loss (MSE), rather than an explicit reconstruction loss.`
  Replace with: `During the forecasting stage, the pretrained MAE mainly serves as a reconstruction-oriented visual feature extractor. The forecasting framework is optimized with the prediction loss, while no additional reconstruction loss is jointly imposed as an auxiliary objective in this stage.`

## Recommended Equations

The previous formula set based on

```text
omega = 1 / (1 + l_rec)
z_v^* = phi([(1-omega) z_vlm ; omega z_rec])
```

is more suitable for a **dual-path adaptive fusion variant**, not for the main `Reconstruction-Oriented MAE Visual Encoder` subsection. For Part 1, the equations should instead focus on:

1. time-series-to-image construction,
2. masked MAE reconstruction,
3. reconstruction-conditioned feature extraction.

The following equation group is better aligned with the current method narrative:

```latex
\begin{gather}
\mathbf{F}_{b,t,d,:}=
\left[
x_{b,t,d},
\left|\mathcal{F}(\mathbf{x}_{b,:,d})\right|_{t},
\sin\!\left(\frac{2\pi t}{P}\right),
\cos\!\left(\frac{2\pi t}{P}\right)
\right] \in \mathbb{R}^{4} \\
\mathbf{I}=
\operatorname{Resize}\!\Big(
\phi_{2D}\big(
\operatorname{Reshape}(\phi_{1D}(\mathbf{F}))
\big)
\Big)
\in \mathbb{R}^{B \times C \times H \times W} \\
(\mathbf{L},\mathbf{M},\Pi)=\operatorname{MAE}_{enc}(\mathbf{I};\rho),
\qquad
\hat{\mathbf{I}}=\operatorname{MAE}_{dec}(\mathbf{L},\Pi) \\
\tilde{\mathbf{z}}_{\mathrm{rec}}
=
\operatorname{GAP}\!\left(\hat{\mathbf{I}} \odot \mathbf{M}_{img}\right),
\qquad
\mathbf{z}_v=\psi\!\left(\tilde{\mathbf{z}}_{\mathrm{rec}}\right)\in\mathbb{R}^{B \times H}
\end{gather}
```

## How to Introduce and Explain the Equations

The equations should not be attached without explanation. To preserve your existing writing style, use a short paragraph before the equations and another short paragraph after them.

Recommended pre-equation transition:

```text
Formally, given a normalized multivariate time series, we first construct image-oriented features by combining raw values, spectral magnitude, and periodic priors, and then map them to a visual tensor. The resulting image is processed by the MAE encoder-decoder to derive reconstruction-conditioned visual features.
```

Recommended post-equation explanation:

```text
In the above formulation, $x_{b,t,d}$ denotes the normalized value of variable $d$ at time step $t$, and $\mathcal{F}(\cdot)$ denotes the Fourier transform along the temporal dimension. The learnable mappings $\phi_{1D}$ and $\phi_{2D}$ transform the constructed temporal features into an image representation. The MAE encoder processes the masked image with masking ratio $\rho$, and the decoder reconstructs the missing content. Finally, the reconstructed regions indicated by the mask are aggregated and projected to obtain the reconstruction-oriented visual embedding $z_v$, which is then used by the downstream fusion module.
```

This structure is recommended:

1. One short transition sentence before the equations.
2. The equation block.
3. One short explanatory paragraph after the equations.

This is much better than attaching the equations without textual explanation.

## Symbol Explanation

You may define the symbols as follows:

- `x_{b,t,d}`: the normalized value of variable `d` at time step `t` in sample `b`
- `mathcal{F}(.)`: Fourier transform applied along the temporal dimension
- `P`: prior period used to encode periodic information
- `phi_{1D}(.)`: learnable 1D mapping over the constructed per-variable temporal features
- `phi_{2D}(.)`: learnable 2D convolutional mapping for image formation
- `rho`: masking ratio of the MAE encoder
- `mathbf{L}`: latent tokens from the masked MAE encoder
- `mathbf{M}`: patch-level binary mask returned by MAE
- `Pi`: restoration index used by the decoder
- `hat{mathbf{I}}`: reconstructed image recovered by the MAE decoder
- `mathbf{M}_{img}`: patch mask expanded to image space
- `psi(.)`: feature projection or enhancement mapping used to obtain the final visual embedding
- `mathbf{z}_v`: reconstruction-oriented visual embedding used by the downstream fusion module

## Optional Sentence for the Architecture Transition

If you want to keep the connection to the full multimodal architecture, insert the following sentence at the beginning or end of the subsection:

```text
Although the complete model is built within a VLM-inspired multimodal framework, the present subsection focuses only on the visual encoder, where the conventional visual backbone is replaced by a reconstruction-oriented MAE module.
```

## What Should Be Removed or Moved Out

The following content is not recommended in Part 1 and should be removed or moved to later subsections:

- `The VLM encoder then passes the visual embedding ... and the placeholder text ... to the fusion module.`
- `In this MAE configuration, the text embedding z_t is a zero-placeholder ...`
- any formula centered on adaptive fusion weight `omega` unless this subsection is explicitly about the dual-path variant

These belong more naturally to the later multimodal fusion/interface subsection.

## Recommended Final Structure for Part 1

To keep most of your current text while improving clarity, organize the subsection in the following order:

1. One opening sentence stating that the full framework is VLM-inspired but this part focuses on the visual branch.
2. One paragraph explaining why MAE is suitable for time-series-derived images.
3. One paragraph introducing the TS-to-Image construction and the reconstruction-oriented feature extraction process.
4. The equation block with a short introduction and explanation.
5. One closing sentence stating that the resulting $z_v$ is used by the downstream multimodal fusion module.

## If You Want a Dual-Path Variant Paragraph

If you still want to mention dual-path briefly without making it the main storyline of Part 1, use the following conservative wording:

```text
As an optional extension, a dual-path variant can further combine standard MAE features and reconstruction-conditioned features through a path-level fusion module. However, in the main formulation of Part 1, we focus on the reconstruction-oriented MAE visual encoder itself.
```
