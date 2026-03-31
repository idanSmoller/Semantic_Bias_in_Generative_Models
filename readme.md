# Semantic Bias in Generative Models

## Multi-Model Experiments

To reproduce the multi-model experiments, execute the following command:

```bash
python multi_model_runner.py --models sdxl sd15 sd21
```

The `--models` argument accepts any combination of the available models. You may specify a subset of models if desired. Additionally, the professions used in the experiments can be customized by modifying the script directly.

## Model Comparison

To perform a comparative analysis between models, use the following command:

```bash
python model_comparison.py --models sdxl sd21
```

This script will generate a comparison between the specified models. Note that this analysis requires the output from the multi-model experiments to be available. Please ensure that `multi_model_runner.py` has been executed prior to running the model comparison.

## Text Projection Analysis

To reproduce the text projection results, run the following command:

```bash
python projection_text.py
```

As with the other scripts, the professions analyzed can be customized by editing the script configuration.

## Gender Bias Identification and Baseline Comparison

### Step 1: Run Gender Bias Identification

Before running the baseline comparison, you must first execute the identification script to generate the required bias analysis CSV files:

```bash
python identification.py
```

This script will:
- Generate images for each profession with male, female, and neutral prompts
- Capture activations from multiple network layers and timesteps
- Compute bias metrics and save them to `generated_images/bias_analysis.csv` and `generated_images/text_bias_analysis.csv`
- Export steering vectors for intervention experiments

**Note:** This script requires significant GPU memory and computational resources.

### Step 2: Run Baseline Comparison

After the identification script completes, you can compare the bias results against the baseline statistics:

```bash
python baseline_comparison.py
```

This analysis correlates the identified biases with baseline gender distribution statistics for each profession. The output includes:
- Correlation statistics (Pearson and Spearman coefficients)
- Heatmaps showing bias patterns across layers and timesteps
- Scatter plots comparing activation-based bias to baseline statistics
- Aggregated profession-level metrics

## Bias Mitigation Experiments

To reproduce the bias mitigation results using vector injection, run:

```bash
python mitigation.py
```

This script will:
- Extract robust gender bias vectors through multi-seed averaging
- Generate images with varying injection strengths to demonstrate bias steering
- Create progression plots showing the effect of different strength values
- Produce bias heatmaps across multiple network layers and timesteps