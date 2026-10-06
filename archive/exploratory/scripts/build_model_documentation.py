"""Build a source-verified, line-by-line guide to the current four models."""
import ast
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import tokenize
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = Path.home() / 'Downloads/ConvRML_Model_Code_Documentation'

# Each entry explains exactly one nonblank source line, in source order.
NOTES = {
'models/shamus_rml.py': '''Module description: the reconstruction backbone follows the verified Shamus FPN architecture.
Import PyTorch for tensors and normalization buffers.
Import neural-network layers and the Module base class.
Import functional operations, used here for resizing feature maps.
Import timm, which supplies the ConvNeXtV2-Base encoder implementation.
Define a reusable 256-channel refinement block.
Construct the block without configurable arguments.
Initialize PyTorch module registration.
Start a sequence with a spatial 3×3 convolution, eight-group normalization, and GELU. Padding preserves height and width.
Add a second convolution–normalization–GELU sequence and close the container. Each group contains 32 channels.
Define the block's forward operation on an N×256×H×W tensor.
Apply both convolutional stages and preserve the input tensor shape.
Define the shared reconstruction network used by all four formal methods.
Construct the backbone and decoder.
Initialize PyTorch module registration.
Create the four-stage ConvNeXtV2-Base feature extractor. pretrained=False means no pretrained weights are downloaded or loaded here, despite the model-name suffix.
Project encoder channels 128, 256, 512, 1024 to 256 using four learned 1×1 convolutions.
Create one refinement block for each of the four FPN scales.
Upsample the fused quarter-resolution map by two, then refine it at half resolution.
Upsample again by two, then refine at full resolution.
Create the ordinary 3×3 RGB residual head: 256 channels become three, with unchanged spatial size.
Register the RGB normalization mean as a movable, nontrainable buffer; persistent=False excludes it from saved state dictionaries.
Register the matching RGB standard deviations, broadcastable over batch and spatial dimensions.
Define feature extraction; the return value is half-resolution decoder features, not the encoder's raw stage list.
Normalize RGB measurements and compute the four encoder feature maps.
Project and refine the deepest encoder stage, starting top-down FPN fusion.
Visit the three shallower stages in reverse depth order.
Resize the current decoder map to the current encoder resolution, add its lateral projection, and refine. Addition requires matching 256-channel shapes.
Return half-resolution features after the first final upsampling block.
Define reconstruction from a measurement and already-computed half-resolution features.
Upsample features to full resolution, predict an RGB correction, and add the original measurement. There is no sigmoid or clipping.
Define the standard model(x) entry point.
Compute features and reconstruct the RGB output.''',
'models/formal_quantile.py': '''Describe the two final-feature-map quantile variants used in the formal comparison.
Import deep copying so the quantile head starts with equal values but independent parameters.
Import tensor creation, gradient contexts, and pinball operations.
Import PyTorch neural-network layers.
Import the shared backbone symbol; this module does not directly instantiate it.
Import the scalar-quantile conditioning block from the larger legacy ConvNeXt source file.
Set the raw endpoint levels to 0.025 and 0.975. The recent 90% report recalibrates these predictions; it does not change this constant to 0.05/0.95.
Define pinball loss for predictions p, targets x, and a per-image quantile vector q.
Compute signed error target minus prediction; reshape q to N×1×1×1 for RGB/spatial broadcasting.
Average max(q·error, (q−1)·error) over batch, channels, and pixels. This is asymmetric absolute loss.
Define a wrapper with an ordinary reconstruction branch and an uncertainty branch.
Accept an existing backbone and either fixed or conditioned mode.
Register the backbone and remember the selected mode.
Select fixed endpoint regression.
Freeze every backbone parameter and put it in evaluation mode. The training script supplies the trained normal checkpoint first.
Create a six-channel 3×3 head: three channels for each of two quantile endpoints.
Disable gradients while copying initialization values.
Repeat the ordinary RGB head weights twice so both endpoint heads initially reproduce the reconstruction correction.
Repeat the three biases for the two endpoint groups.
Select scalar-conditioned quantile regression.
Use 256 feature channels, a 64-dimensional quantile embedding, and a 16-channel broadcast condition map.
Copy the reconstruction head into a separate trainable RGB quantile head. The base remains trainable in this mode.
Reject unsupported modes.
Override train() to keep a frozen backbone in evaluation mode.
Apply the requested training/evaluation state to the wrapper and descendants.
Force the backbone back to evaluation mode for the fixed variant.
Return self, following the standard PyTorch train() interface.
Compute shared full-resolution features and the ordinary reconstruction.
Handle the frozen-backbone variant.
Without recording gradients, run the decoder to full resolution and reconstruct r=y+head(f).
For conditioned mode, compute the same tensors with gradients so both objectives can update the backbone.
Return f of shape N×256×H×W and r of shape N×3×H×W.
Predict quantiles from a measurement and a precomputed feature map; q is required only in conditioned mode.
In fixed mode, reshape six output channels into N×2×3×H×W and add y to both endpoint residuals.
In conditioned mode, modulate features using q, apply the RGB quantile head, and add the measurement.
Return a reconstruction and the two raw endpoint predictions.
Run the shared feature computation once.
For fixed mode, evaluate the six-channel head once and return its two RGB groups.
For conditioned mode, evaluate the lightweight conditioned branch at each configured endpoint, reusing f. Quantile ordering is not enforced.''',
'baselines/formal_rml.py': '''Describe the full-backbone baseline models, distinct from older head-only toy baselines.
Import torch for output concatenation.
Import layers and initialization utilities.
Import the shared FPN reconstruction backbone.
Subclass the ordinary backbone for MC dropout or an ensemble member.
Accept the desired method.
Build the entire backbone and remember the method.
Reject any method except ensemble or mc.
Choose the ensemble-member architecture.
Add a 3×3 RGB raw-variance head operating on the same full-resolution features as the mean head.
Start variance weights at zero and biases at −4.6. The wrapper later uses softplus, initially yielding variance approximately 0.01.
Choose the MC-dropout architecture.
Explain that dropout actually participates in decoder computation during training and stochastic inference.
Visit all four FPN refinement blocks and both upsampling refinement blocks.
Start a replacement layer list for this block.
Iterate over the original block's registered layers.
Preserve each original layer in order.
Insert Dropout2d(p=0.1) after every GELU. It drops feature channels spatially; two sites per block give twelve decoder dropout sites.
Replace the block's sequence with the augmented sequence; registration includes the dropout modules.
Define the baseline forward pass.
Compute full-resolution decoder features and the ordinary residual RGB mean.
For an ensemble member, concatenate mean and raw variance to N×6×H×W. The second half is not yet a positive variance.
For MC dropout, return just the RGB prediction.''',
'baselines/common.py': '''Import PyTorch for distribution quantiles and tensor aggregation.
Define a symmetric Gaussian moment interval with nominal miscoverage alpha.
Document that this construction alone does not ensure empirical coverage.
Reject alpha outside the open interval (0,1).
Find the standard-normal 1−alpha/2 quantile on the mean tensor's device/dtype. At alpha=0.1 this is approximately 1.645.
Clamp variance below at zero, take its square root, and multiply by the Gaussian multiplier.
Return lower and upper symmetric endpoints.
Aggregate predictions shaped members×batch×channels×height×width.
Check matching shapes, five dimensions, and at least one member.
Raise an informative error if the aggregation contract is violated.
Average member means to obtain the ensemble reconstruction.
Average members' predicted variances: the learned within-member/aleatoric component.
Compute population variance of member means (denominator M): the between-member/epistemic component.
Return mean, total variance, and its two components. Total variance is their sum.''',
'baselines/mc_dropout.py': '''Identify the image-only MC-dropout adapter and its source lineage.
Import the helper for implementing a with-context.
Import torch for sampling and tensor operations.
Import Module and dropout layer types.
Import the uncalibrated Gaussian interval helper.
Convert the following generator into a context manager.
Define a temporary inference mode with dropout active.
Record every module's existing training flag for exact restoration.
Start a protected block so restoration also occurs after exceptions.
Put the entire model in evaluation mode first.
Visit every registered module.
Enable only recognized dropout layers. Other layers, including normalization, remain in evaluation mode.
Run the caller's inference code inside this temporary state.
Always execute cleanup when leaving the context.
Restore each original flag directly, avoiding recursive train() calls that could overwrite child states.
Define the reusable stochastic-prediction wrapper.
Accept the network, number of MC passes, optional constant observation variance, and nominal interval alpha.
Initialize module registration.
Require at least two configured MC samples and nonnegative observation variance.
Register the base model as a submodule.
Store the default sample count.
Store the optional variance term; the formal experiment uses zero additional observation variance.
Store the alpha used only by the wrapper's uncalibrated Gaussian interval.
Define a reconstruction convenience method with a requested number of samples.
During ordinary one-pass training, call the base network directly and retain gradients.
Otherwise draw stochastic predictions and average across the sample dimension. predict_samples disables gradients.
Disable gradient recording for sample inference.
Define explicit stochastic sample generation.
Resolve the requested sample count or use the configured default.
Reject an empty sample request.
Activate dropout while keeping the rest of the model in evaluation mode.
Run n forward passes and stack results into n×N×3×H×W; this uses memory proportional to n.
Disable gradient recording for moment estimation.
Define memory-efficient stochastic mean/variance estimation.
Explain that the following online update avoids retaining all sampled images.
Activate the temporary dropout inference state.
Initialize the running mean and squared-deviation accumulator as absent.
Draw the configured number of predictions.
Evaluate one stochastic realization of the network.
Initialize the mean from the first sample and the squared-deviation sum to zero.
Handle subsequent samples.
Use Welford's online update for the mean and sum of squared deviations, avoiding subtraction of large raw moments.
Divide the accumulated squared deviations by T to obtain population dropout variance, not the T−1 sample estimator.
Return the sampled mean and variance plus the configured observation variance.
Define the wrapper's interval-producing forward entry point.
Compute stochastic moments.
Turn moments into a nominal Gaussian interval. The separately calibrated report instead uses fitted radius factors.
Concatenate lower, mean, upper into N×9×H×W. The formal training script uses its own equivalent online inference loop.''',
'baselines/deep_ensemble.py': '''Describe the RGB Gaussian-negative-log-likelihood ensemble adapter.
Import torch for stacking and concatenation.
Import Module and ModuleList.
Import softplus from the functional API.
Import the total-variance aggregator and nominal Gaussian interval helper.
Wrap one network that emits RGB means and raw variances.
Accept the base network, number of image channels, and a positive variance floor.
Register the network and save channel count and variance floor.
Reject nonpositive variance floors.
Define conversion of raw model output to valid Gaussian parameters.
Evaluate the underlying six-channel model.
Require exactly C mean channels and C variance channels.
Split the tensor and apply softplus plus 1e−6 by default to keep predicted variance positive.
Define the member training loss.
Obtain mean and positive variance.
Compute average Gaussian NLL without the parameter-independent log(2π)/2 constant. This balances squared error against predicted variance.
Define a wrapper around independently trained ensemble members.
Accept members and the nominal interval alpha.
Register members so their parameters and device moves are tracked, and store alpha.
Require at least one member.
Define ensemble mean and total variance inference.
Evaluate each member, obtaining one mean/variance pair per member.
Stack member tensors and use the law of total variance; discard the separately returned component tensors here.
Return the ensemble reconstruction and total variance.
Provide a shortcut returning only the ensemble mean.
Define the interval-producing forward pass.
Compute ensemble moments.
Build a nominal Gaussian interval. Independent empirical calibration is applied outside this wrapper for the reported results.
Concatenate lower, mean, and upper RGB tensors into nine channels.''',
'models/convnext.py:QuantileConditioner': '''Define the scalar-q modulation block imported by the current formal quantile model.
Begin the block's description: it conditions spatial image features on a scalar quantile.
Introduce the three conditioning mechanisms.
Describe an MLP embedding and FiLM scale/shift, then introduce the gated path.
Describe spatial broadcasting of the quantile condition and the residual path.
State that modulation begins as the identity, preserving the copied reconstruction initially.
Explain the initialization's intended stability; this is a design motivation, not an accuracy guarantee.
Close the class documentation string.
Accept feature width and optional embedding/condition widths. The formal caller overrides defaults with 256, 64, 16.
Initialize module registration.
Store the feature-channel count.
Begin the quantile embedding MLP.
Project each scalar q into the embedding dimension: N×1 to N×64 in the formal model.
Apply the smooth SiLU activation.
Apply a second learned linear projection within the embedding dimension.
Apply another SiLU activation.
Close the embedding network.
Map each embedding to twice the feature width, holding one scale and one shift per channel.
Map the embedding to the broadcast condition channels (16 in the formal model).
Create a 1×1 convolution mixing the 256 feature channels and 16 condition channels back into 256 channels.
Map the embedding to one gate logit per feature channel.
Document why modulation is initialized to preserve incoming features.
Document that a separately copied RGB head can subsequently specialize into quantile regression.
Zero FiLM projection weights so its initial scales and shifts do not depend on q.
Zero FiLM biases, giving exactly zero scale offsets and shifts.
Zero residual-convolution weights so the residual output starts at zero.
Zero residual-convolution biases.
Zero gate weights so initial gate logits are zero.
Zero gate biases; sigmoid(0)=0.5, but the residual is zero, so total modulation is still the identity.
Define modulation of features N×C×H×W by per-image q.
Accept a one-dimensional quantile batch.
Insert a singleton dimension, giving N×1.
Check that the normalized quantile shape is two-dimensional with one scalar per image.
Reject unsupported quantile shapes and report the supplied shape.
Check that the quantile and feature batch sizes agree.
Reject mismatched batches. This method does not check q is in [0,1] or move q to the feature device.
Cast q to the feature dtype and compute its embedding. The caller must already place q on the correct device.
Split the FiLM output into C scales and C shifts.
Add singleton spatial dimensions to scale for broadcasting.
Add singleton spatial dimensions to shift for broadcasting.
Apply FiLM: f_film=f·(1+scale)+shift, preserving N×C×H×W.
Project the embedding into condition channels and introduce singleton spatial dimensions.
Broadcast the condition vector across the image without explicitly repeating storage at this step.
Concatenate conditioned features and the broadcast condition, then apply the learned 1×1 residual convolution.
Apply sigmoid to obtain gates in (0,1), broadcast spatially.
Return the FiLM features plus the gated residual. This does not enforce monotonicity of predictions as q increases.'''
}

OVERVIEW = '''# ConvRML: current four-model code guide

This package documents the project-owned model code used in the latest real-image four-method comparison. Each physical source line in the seven documented units has an explanation, including blank lines and multi-statement lines. Original line numbers refer to the source snapshot packaged here. Annotated copies add comments only; their Python syntax trees are verified against the originals.

## What is included

- Shared ShamusRML ConvNeXtV2-Base FPN reconstruction architecture.
- Fixed-head im2im-style adaptation and scalar-Q-conditioned ConvRML.
- Full-network MC-dropout and deep-ensemble architectures, wrappers, and moment helpers.
- The QuantileConditioner class imported from models/convnext.py, with its original line numbers. Other classes in that file belong to older architectures and are not the backbone used in this comparison.
- Unmodified training and evaluation source snapshots as supporting references. These scripts and third-party timm/PyTorch internals are not part of the line-by-line model annotations.

## Data flow and shapes

Notation: y is the RGB measurement; x is its paired target; N is batch size. The formal images are N×3×256×256 in [0,1]. The backbone method names use x for the input measurement in places; that local variable should not be confused with target x in the losses.

The encoder receives normalized y and returns four stages: N×128×64×64, N×256×32×32, N×512×16×16, N×1024×8×8. Four 1×1 lateral convolutions project these to 256 channels. Top-down resize, addition, and refinement yield a quarter-resolution map. up1 produces N×256×128×128; up2 produces N×256×256×256. The normal reconstruction is r=y+Conv3×3(f). Outputs are unconstrained real numbers; clipping in the example figures is for display only.

## How the four methods differ

1. Fixed head (im2im-style): first train the normal backbone on MSE. Freeze it and train a six-channel 3×3 head on the last feature map. Its channel groups predict two RGB quantile residuals added to y. This is our ConvRML adaptation, not a claim of exact reproduction of an external im2im architecture.
2. Q-conditioned: train a normal reconstruction branch and a separate conditioned RGB head jointly. A scalar q per image passes through an MLP, FiLM scale/shift, and a gated broadcast-condition residual. Shared features are computed once; predicting another q reruns the conditioner and quantile head. The formal model uses embedding width 64 and condition width 16. The head can be evaluated at arbitrary q, but the code does not guarantee monotonic quantiles or an exact calibrated conditional distribution.
3. Deep ensemble: train five independent full backbones, each predicting a mean and positive variance. Reconstruction is the mean of five means. Total variance = mean predicted variance + population variance of predicted means. These components are commonly interpreted as aleatoric and epistemic proxies.
4. MC dropout: train one full backbone with channel dropout p=0.1 after decoder GELUs. At inference, enable dropout and take 30 stochastic passes. Reconstruction is their average; variance is their population variance. The formal experiment adds no observation-noise variance term.

## Losses and training behavior

Pinball loss: rho_q(x−p)=max(q(x−p),(q−1)(x−p)), averaged over images, RGB channels, and spatial pixels.

Normal and MC training use mean squared error. Fixed-head training averages the two endpoint pinball losses with the base frozen. Q-conditioned training samples q~Uniform(0,1) independently per image, uses the same q for all RGB pixels of that image, and minimizes MSE(r,x)+pinball(p_q,x,q), with equal coefficients. Q-conditioned checkpoint selection uses the average endpoint pinball loss, not the full training objective. Ensemble members minimize 0.5·mean(log(v)+(x−mu)^2/v), with v=softplus(raw_v)+1e−6.

The shared 25,000-pair manifest contains 20,000 training, 2,500 validation, and 2,500 test images. The latest table uses the first 1,000 test entries, shared across methods. The models were trained through 20 epochs with best-validation checkpoint selection; a selected checkpoint can precede epoch 20. Fixed-head training has a separate normal-model stage, so its total training budget differs. The formal full models initialize with pretrained=False; they do not load Shamus's trained parameters. AdamW uses learning rate 1e−4, weight decay 1e−4, batch size 2, bfloat16 autocast, and gradient clipping at norm 1. Two-epoch job chunks resume model, optimizer, and RNG state. Five ensemble workers and one MC worker are independent jobs, not six-way data parallelism for a single model.

## Calibration in the latest alpha=0.1 report

Alpha=0.1 means a 90% target for average pixel-channel coverage. The raw fixed/conditioned endpoint levels remain 0.025 and 0.975. The report adjusts their widths with a factor fitted on 2,048 separate calibration images; it does not rerun quantile inference at 0.05/0.95.

For fixed/conditioned models: wL=max(r−p_low,0.001), wU=max(p_high−r,0.001). The calibration score is max((r−x)/wL,(x−r)/wU). A single global factor lambda is the empirical 90th percentile with the higher interpolation rule. Reported bounds are [r−lambda·wL,r+lambda·wU]. These floors ensure positive widths around the reconstruction even if raw quantiles cross it.

For ensemble/MC: s=max(sqrt(v),0.0001). Calibrate scores |x−mu|/s, then use [mu−lambda·s,mu+lambda·s]. This fitted factor replaces the Gaussian multiplier; do not multiply by 1.645 again. Factors in the latest report: fixed 0.78355944, conditioned 0.90196043, ensemble 1.26801300, MC 3.61313987.

This is pooled empirical pixel calibration. Spatially correlated pixels are not independent calibration images, and this procedure is not a per-image or distribution-free 90% guarantee. Calibration changes intervals, not reconstructions, PSNR, or SSIM.

## Reading the latest metrics

PSNR per image = −10 log10(mean squared RGB error), using data range 1. The report averages these 1,000 dB values; it is not PSNR calculated from the pooled MSE. SSIM uses the TorchMetrics implementation with data_range=1 and is averaged over images. Coverage is the fraction of RGB target values inside the inclusive bounds, averaged over images. Width is mean(upper−lower) in normalized intensity units. Unclipped predictions are used for all metrics.

## Using this package

Open model_code_guide.html in a browser for the searchable, printable line-by-line guide. The Markdown version contains the same explanations. originals/ preserves the actual source snapshot. annotated/ contains comment-annotated copies of complete model files; the conditioner is provided as a labeled class excerpt. source_manifest.json records SHA-256 hashes and documented ranges. supporting/ includes the training/evaluation scripts and latest measurements. This is a documentation/source package, not a self-contained training environment or checkpoint bundle.

## Line-by-line reference
'''

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md = [OVERVIEW]
    manifest = []
    for key, notes in NOTES.items():
        rel = key.split(':')[0]
        text = (ROOT / rel).read_text()
        lines = text.splitlines()
        start, end = 1, len(lines)
        if ':' in key:
            node = next(n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == 'QuantileConditioner')
            start, end = node.lineno, node.end_lineno
        segment = lines[start-1:end]
        explanations = notes.splitlines()
        assert len(explanations) == sum(bool(s.strip()) for s in segment), (key, len(explanations), sum(bool(s.strip()) for s in segment))
        it = iter(explanations)
        entries = [(i, line, next(it) if line.strip() else 'Blank separator; no computation or state change.') for i,line in enumerate(segment,start)]
        dest = OUT/'originals'/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(text)
        manifest.append({'file':rel,'documented_start':start,'documented_end':end,'sha256':hashlib.sha256(text.encode()).hexdigest(),'lines_documented':len(entries)})
        md.append(f'\n### {key}\n')
        for i,line,note in entries:
            md.append(f'\n**Line {i}.** {note}\n\n```python\n{line}\n```\n')
        # Insert comments before physical lines, grouping explanations before a
        # multiline string so no comment text becomes part of a docstring.
        snippet='\n'.join(segment)+'\n'
        owners={}
        for token in tokenize.generate_tokens(io.StringIO(snippet).readline):
            if token.type==tokenize.STRING and token.start[0]!=token.end[0]:
                for row in range(token.start[0],token.end[0]+1):owners[row]=token.start[0]
        comments={}
        for local,(i,line,note) in enumerate(entries,1):
            owner=owners.get(local,local)
            comments.setdefault(owner,[]).append(f'# Original L{i}: {note}')
        annotated=[]
        for row,line in enumerate(segment,1):
            indent=line[:len(line)-len(line.lstrip())]
            annotated.extend(indent+c for c in comments.get(row,[]))
            annotated.append(line)
        result='\n'.join(annotated)+'\n'
        assert ast.dump(ast.parse(snippet))==ast.dump(ast.parse(result)),key
        target=OUT/'annotated'/rel
        if ':' in key:target=target.with_name('quantile_conditioner_excerpt.py')
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(result)
    body='\n'.join(md)
    (OUT/'model_code_guide.md').write_text(body)
    # A small deterministic renderer keeps this deliverable dependency-free.
    parts=[];code=False;block=[]
    for line in body.splitlines():
        if line.startswith('```'):
            if code:parts.append('<pre><code>'+html.escape('\n'.join(block))+'</code></pre>');block=[]
            code=not code;continue
        if code:block.append(line);continue
        if line.startswith('#'):
            level=len(line)-len(line.lstrip('#'));parts.append(f'<h{level}>'+html.escape(line[level:].strip())+f'</h{level}>')
        elif line.strip():parts.append('<p>'+html.escape(line).replace('**','')+'</p>')
    css='body{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.6 system-ui;color:#172535}h1,h2,h3{color:#135d73}h2{margin-top:40px}h3{border-top:3px solid #135d73;padding-top:24px}pre{background:#edf3f6;padding:12px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px;border-left:3px solid #64a4b8}p{margin-bottom:8px}@media print{body{margin:0;font-size:10pt}h3{break-before:page}pre{break-inside:avoid}}'
    (OUT/'model_code_guide.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>ConvRML model code guide</title><style>'+css+'</style><body>'+'\n'.join(parts)+'</body></html>')
    (OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    for rel in ['scripts/formal_rml_train.py','scripts/formal_quantile_train.py','scripts/formal_quantile_evaluate.py','scripts/calibrate_formal95.py','scripts/evaluate_four_1000_alpha10.py']:
        target=OUT/'supporting'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/rel,target)
    for name in ['summary.md','summary.json','calibration.json','per_image.csv']:
        shutil.copy2(OUT.parent/'ConvRML_1000_Test_Alpha_0p1'/name,OUT/'supporting'/name)
    (OUT/'README.txt').write_text('Start with model_code_guide.html (browser) or model_code_guide.md.\nAll current project-owned model implementation lines are explained; see scope in the guide.\nAnnotated copies preserve the original Python AST. Production model files were not modified.\n')
    with zipfile.ZipFile(OUT.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(OUT.parent))
    print(json.dumps({'directory':str(OUT),'documented_lines':sum(x['lines_documented'] for x in manifest),'units':len(manifest),'syntax_tree_checks':'all passed'},indent=2))

if __name__=='__main__':main()
