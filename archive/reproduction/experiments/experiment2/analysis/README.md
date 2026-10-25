# Experiment 2 · LPIPS and uncertainty/error correlation

Evaluate the completed epoch-50 RML and normalized DiffuserCam models: Im2im, five-member ensemble, MC dropout (`p=0.05`, 30 passes), and QUTCC. Reuse each method's saved independent calibration. No training and no fitting on test data.

- **LPIPS:** pretrained AlexNet LPIPS v0.1, via TorchMetrics. RGB images are clipped to [0,1] and `normalize=True` performs the required conversion for LPIPS. Lower is better.
- **Error map:** `mean_RGB(abs(prediction - ground_truth))`.
- **Uncertainty map:** `mean_RGB(upper_bound - lower_bound)`.
- **Pearson / Spearman:** compute across all 256×256 pixels separately within each image, then report image-wise means, medians, standard deviations and image-bootstrap 95% intervals. Constant maps produce undefined correlations and valid sample counts are explicit.
- **Primary:** unsmoothed maps. Gaussian σ=0.5, 1, 2, 4 pixels are secondary sensitivity analyses; no scale is selected using the test set.
- Image-level correlations between mean width and MAE are reported separately from pixel-level correlations.
- Region Pearson/Spearman are omitted. Maps must match exactly; mismatches are rejected instead of silently cropped. Smoothing is applied to 2D RGB-averaged maps and never across color channels.

The saved-example preview uses the first **50** fixed test IDs in manifest order; it is not the full 5,000-image benchmark. RML QUTCC epoch-50 calibration, export, and full-test analysis have now completed; the old epoch-20 checkpoint is not substituted. Earlier preview outputs may retain their historical pending marker. Examples 1, 2 and 6 are shown without selecting for a good correlation. Scatter plots use deterministic display subsampling; correlation coefficients use every pixel. All maps in an example share the same display scale.

`analyze_saved_examples.py` analyzes the saved demo NPZ files; `visualize.py` creates a PDF and PNG figures. `evaluate_full_test.py` is the GPU evaluator for all 5,000 test pairs per camera. Baseline weights are restored from archived GitHub releases with SHA-256 verification into this analysis's temporary `weights/` directory. Full-test evaluations use fresh, documented stochastic MC draws; they need not exactly reproduce the demo's draws. QUTCC uses float32; other models retain their original bfloat16 inference protocol. All image metrics use clipped predictions/endpoints; calibrations were fitted on separate calibration data.

```sh
python test_metrics.py
python analyze_saved_examples.py --output ~/Downloads/ConvRML_Experiment2_Correlation_Analysis
python visualize.py --results ~/Downloads/ConvRML_Experiment2_Correlation_Analysis
# On Unicorn, from the project root, with archived weights restored:
python experiments/experiment2_uncertainty_analysis_20261019/evaluate_full_test.py --camera diffuser --method im2im --smoke
```

Requirements: Python 3.11 for Unicorn; PyTorch, torchvision, torchmetrics, scipy, numpy, matplotlib. LPIPS needs the pretrained AlexNet weight file in the Torch cache. Reference: [original LPIPS implementation](https://github.com/richzhang/PerceptualSimilarity).

Correlation quantifies association with realized reconstruction error. It does not establish interval calibration, equal accuracy, causality, or usefulness beyond the evaluated image distribution. Correlation differences and LPIPS differences should be assessed with paired image resampling before claiming a significant improvement.


## Curves at every five epochs

`evaluate_full_test.py --epoch 5` (and 10, …, 50) evaluates the same 5,000 held-out test images. Epoch 50 retains the existing result location and includes secondary smoothing scales; other milestones compute the primary unsmoothed correlations. Calibration factors or quantile levels are read from that exact checkpoint's completed calibration report.

`milestone_pipeline.py` runs on the Mac with its existing GitHub/SSH authentication. It restores one milestone's archived baseline weights at a time, checks SHA-256, transfers them to the analysis-only Unicorn weights folder, and submits eight independent model/camera evaluations. Baseline jobs continue after their corresponding existing epoch-50 evaluations. Earlier QUTCC milestones use their already completed calibrations directly, so they do not wait for final epoch-50 calibration. Epoch-50 work is reused. After all eight results are complete and their test identities match within each camera, only the temporary restored copies for that milestone are removed. Original checkpoints and GitHub archives are preserved. The next milestone can then start. Maximum additional live checkpoint storage is one milestone (about 5 GB), and there are eight evaluation streams.

LPIPS, mean per-image Pearson and mean per-image Spearman are plotted against **5, 10, 15, 20, 25, 30, 35, 40, 45, 50**. Full-test points and the 50-image preview are separate. Missing points remain pending; there is no interpolation through unevaluated epochs. Region-based metrics are excluded.

## Source publication

See the [complete workflow](../../../archive/README.md) and [metric definitions](../../../archive/README.md). The readable files preserve the captured Python behavior; [source provenance](../../../archive/README.md) explains historical resume hashes. For reconstruction/width/coverage CSV and PNG curves from existing reports, use [`tools/plot_learning_curves.py`](../../../archive/README.md). The Mac-only publishers require the separate demo checkout and existing local archive manifests; running the evaluator alone does not deploy a website.
