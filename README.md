# Uncertainty-Aware Diabetic Retinopathy Detection  
### Deep-Learning + Bayesian Inference • ADS Bayesian ML Final Project

> *“A model that tells us when **not** to trust it is often worth more than a model that is occasionally wrong but always confident.”*

---

## 📋 Table of Contents  
1. [Project Motivation](#project-motivation)  
2. [Quick Start](#quick-start)  
3. [Data Pipeline](#data-pipeline)  
4. [Model Architectures](#model-architectures)  
5. [Training Details](#training-details)  
6. [Uncertainty Estimation](#uncertainty-estimation)  
7. [Training Results](#results--visualisations)  
8. [Inference Summary for Final Model (CNN + 1 Bayes)](inference-summary)
9. [Classification Report (Excluding IDK Cases)](classification-report)
10. [Re-Use in Production](#re-use-in-production)  
11. [Key Paper & Further Reading](#key-paper--further-reading)  
12. [Team & Acknowledgements](#team--acknowledgements)

---

## Project Motivation  
Diabetic Retinopathy (DR) is a leading cause of preventable blindness. Traditional CNNs **over-confidently** mis-classify rare or ambiguous fundus images, limiting their clinical adoption.  
Our solution marries a strong convolutional backbone with **Bayesian linear layers** and **Monte-Carlo (MC) dropout** to provide:

- **Well-calibrated probabilities**  
- An automatic *“I-don’t-know”* (IDK) mechanism that defers uncertain cases to specialists  
- A light-weight PyTorch implementation ready for edge deployment

---

## Quick Start  

```bash
# 1 ▪ Clone & install
git clone https://github.com/your-org/bayes-dr.git && cd bayes-dr
conda env create -f environment.yml        # OR: pip install -r requirements.txt
conda activate bayes-dr

# 2 ▪ Download balanced subset (700 imgs × 5 classes)
python scripts/download_data.py --kaggle-dataset tanlikesmath/diabetic-retinopathy-resized

# 3 ▪ Train baseline Bayesian-ResNet
python src/train.py --model bayes_resnet --epochs 15

# 4 ▪ Launch interactive dashboard
jupyter lab notebooks/01_explore_results.ipynb
```

## Data Pipeline  

| Step                  | Tool                                                         | Output                |
|-----------------------|--------------------------------------------------------------|-----------------------|
| **Download**          | `kagglehub`                                                  | `data/raw/`           |
| **Sub-sampling**      | `pandas` + `shutil`                                          | 700 images per class  |
| **Train / Val split** | PyTorch `random_split` (70 / 30)                             | `data/processed/`     |
| **Augmentation**      | `torchvision.transforms` (resize 224², random H-flip, color jitter) | —                     |

> **Class labels:**  
> 0 Healthy · 1 Mild DR · 2 Moderate DR · 3 Severe DR · 4 Proliferative DR

```bash
# Example: build balanced subset
python scripts/build_subset.py --samples-per-class 700 --seed 111
```



## Model Architectures  

| File                       | Backbone              | Bayesian Head                      |
|----------------------------|-----------------------|------------------------------------|
| `bayes_resnet_trial.py`    | ResNet-18             | 128 → 5 (1 × `BayesianLinear`)     |
| `bayes_resnet_trial1.py`   | ResNet-18             | 128 → 64 → 5 (2 × `BayesianLinear`)|
| `custom_bayes_cnn_try.py`  | 3-block custom CNN    | 128 → 5 (1 × `BayesianLinear`)     |
| `custom_bayes_cnn_5.py`    | 3-block custom CNN    | 128 → 64 → 5 (2 × `BayesianLinear`)|


_All heads expose `sample_elbo()` from [BLiTZ](https://github.com/piEsposito/blitz-bayesian-deep-learning) for variational inference._

```python
@variational_estimator
class BayesianDRNet(nn.Module):
    def forward(self, x):
        features = self.extractor(x)
        logits = self.bayes_classifier(features)
        return logits
```

## Training Details  

```math
\mathcal{L}_{\mathrm{ELBO}}
= \mathbb{E}_{q_\theta(w)}\bigl[\log p(y \mid x, w)\bigr]
- \beta\,\mathrm{KL}\bigl(q_\theta(w)\,\|\,p(w)\bigr)
```

- $$\(\beta = 5\times10^{-3}\)$$ controls posterior complexity.  
- **Optimizer:** Adam (lr 1e-4, weight-decay 1e-5)  
- **Batch size:** 32 • **Epochs:** 15–20  
- **Hardware:** Tesla T4 (16 GB) — ~20 min per run  


## Uncertainty Estimation  

| Symbol                 | Definition                                       |
|------------------------|--------------------------------------------------|
| $$\(p_k\)$$                | Mean softmax probability over $$\(N\)$$ MC samples    |
| $$\(\sigma_k\)$$           | Std-dev of softmax over samples (per class)       |
| **Predictive Entropy** | $$\(\displaystyle \mathcal{H} = -\sum_{k=1}^{K} p_k \log p_k\)$$ |
| **IDK Gate**           | Defer if $$\(\mathcal{H} > 0.6\times10^{-3}\)$$       |

```python
from src.evaluate import mc_predict

cls, mean_p, std_p, H = mc_predict(model, image, n_samples=30)
if H > 6e-4:
    print("→  IDK (refer to ophthalmologist)")
else:
    print(f"→  Predicted class {cls} (confidence {mean_p[cls]:.2%})")

```

## Training Results 

| Model                | Accuracy (after IDK) | 
|----------------------|----------------------|
| ResNet + 1 Bayes     | 0.12                 | 
| ResNet + 2 Bayes     | 0.22                 | 
| CNN + 1 Bayes        | **0.84**             | 
| CNN + 2 Bayes        | 0.15                 | 

![ELBO Curve](docs/figs/elbo.png)  
![Uncertainty vs Error](docs/figs/uncert_scatter.png)

 *Green dots = correct predictions, red = mis-classifications.*

 ## Inference Summary for Final Model (CNN + 1 Bayes)

- **Overall Performance (on hold-out set)**  
  - **Accuracy** on non-IDK predictions: **98.36 %**  
  - **Coverage** (proportion of samples auto-predicted): **60.80 %**  
  - **IDK Rate** (referred to expert): **39.20 %**  

- **Class-wise “I-Don’t-Know” Rates**  
  | True Class      | IDK Rate |
  |-----------------|---------:|
  | Healthy (0)     | 17.8 %   |
  | Mild (1)        | 44.0 %   |
  | Moderate (2)    | 82.5 %   |
  | Severe (3)      | 32.0 %   |
  | Proliferative (4)| 20.7 %  |

- **Per-Class Entropy Thresholds** (80th percentile)  
  | Class | Threshold (entropy) |
  |------:|--------------------:|
  | 0     | 0.0238              |
  | 1     | 0.7656              |
  | 2     | 1.0623              |
  | 3     | 0.5011              |
  | 4     | 0.0001              |

- **Threshold‐Selection Strategies**  
  | Class | Strategy            | Chosen Threshold | TP   | FP  | F1   |
  |------:|---------------------|-----------------:|-----:|----:|-----:|
  | 0     | `min_fp`            | 0.1000           | 137  | 20  | 0.919 |
  | 1     | `min_fp`            | 0.1000           | 122  | 31  | 0.830 |
  | 2     | `hybrid_fp_f1`      | 0.1000           | 106  | 56  | 0.711 |
  | 3     | `strict_tp_entropy` | 0.1000           | 101  | 1   | 0.828 |
  | 4     | `strict_tp_entropy` | 0.1000           | 122  | 4   | 0.917 |


## Classification Report (Excluding IDK Cases)

| Class          | Precision | Recall | F1-score | Support |
| -------------- | --------- | ------ | -------- | ------- |
| Healthy        | 0.944     | 1.000  | 0.971    | 51      |
| Mild           | 1.000     | 1.000  | 1.000    | 33      |
| Moderate       | 1.000     | 0.800  | 0.889    | 10      |
| Severe         | 1.000     | 0.975  | 0.987    | 40      |
| Proliferative  | 1.000     | 1.000  | 1.000    | 49      |
| **Accuracy**   |           |        | **0.984**| 183     |
| **Macro avg**  | 0.989     | 0.955  | 0.970    | 183     |
| **Weighted avg** | 0.985   | 0.984  | 0.983    | 183     |


- **Representative IDK Examples**  
The first six images flagged as IDK tended to be borderline or low-contrast fundus scans, with uncertainties ranging above 0.50—correctly deferring to expert review.

- **Key Takeaway**  
By tuning per-class entropy thresholds and adopting a “refer-on-uncertainty” policy, our CNN + 1 Bayes model achieves **98.36 %** accuracy on confident cases while safely deferring **39.2 %** of ambiguous inputs to specialists.


## Re-Use in Production  

```bash
# Export model via TorchScript
python - <<EOF
import torch
model = torch.load("best_model.pt")
traced = torch.jit.trace(model, torch.randn(1,3,224,224))
traced.save("model_trace.pt")
EOF

# Serve with FastAPI
uvicorn deploy.app:app --host 0.0.0.0 --port 8000
```

## Key Paper & Further Reading  

- **Our report:** “Uncertainty-aware diabetic retinopathy detection using deep learning enhanced by Bayesian approaches”  
- Seitzer et al., “BLiTZ: Bayesian Layers in Torch” (2020)  
- Kendall & Gal, “What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?” (NIPS 2017)  

## Team & Acknowledgements  

**UChicago · Applied Data Science** – Spring 2025  
Angus Ho ·  Anubuthi Kottapalli · Ashwin Ram Venkataraman · Gyujin Seo

Thanks to Prof. Batu for clinical guidance, the Kaggle DR community, and the BLiTZ maintainers.

> **MIT License** © 2025 ADS Bayesian ML Epochalpyse Now Group  
