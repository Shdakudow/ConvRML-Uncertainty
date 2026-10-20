# Experiment 2 · LPIPS and uncertainty/error correlation

Evaluate the completed epoch-50 RML and normalized DiffuserCam models: Im2im, five-member ensemble, MC dropout (`p=0.05`, 30 passes), and QUTCC. Reuse each method's saved independent calibration. No training and no fitting on test data.

- **LPIPS:** pretrained AlexNet LPIPS v0.1, via TorchMetrics. RGB images are clipped to [0,1] and `normalize=True` performs the required conversion for LPIPS. Lower is better.
- **Error map:** `mean_RGB(abs(prediction - ground_truth))`.
- **Uncertainty map:** `mean_RGB(upper_bound - lower_bound)`.
- **Pearson / Spearman:** compute across all 256×256 pixels separately within each image, then report image-wise means, medians, standard deviations and image-bootstrap 95% intervals. Constant maps produce undefined correlations and valid sample counts are explicit.
- **Primary:** unsmoothed maps. Gaussian σ=0.5, 1, 2, 4 pixels are secondary sensitivity analyses; no scale is selected using the test set.
- Image-level correlations between mean width and MAE are reported separately from pixel-level correlations.
- Region Pearson/Spearman are omitted. Maps must match exactly; mismatches are rejected instead of silently cropped. Smoothing is applied to 2D RGB-averaged maps and never across color channels.

The saved-example preview uses the first **50** fixed test IDs in manifest order; it is not the full 5,000-image benchmark. Pending RML QUTCC calibration/export is explicitly marked and never replaced by the old epoch-20 checkpoint. Examples 1, 2 and 6 are shown without selecting for a good correlation. Scatter plots use deterministic display subsampling; correlation coefficients use every pixel. All maps in an example share the same display scale.

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
