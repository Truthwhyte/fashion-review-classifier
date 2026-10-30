# Will they recommend it? Spotting unhappy fashion shoppers from review text

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Truthwhyte/fashion-review-classifier/blob/main/fashion_review_classifier.ipynb)

An online clothing retailer gets thousands of reviews. This project asks whether we can automatically flag the reviews whose writer would **not** recommend the item, so a merchandising team sees problems such as bad sizing or poor fabric early.

I compare three approaches on the same held-out test set:

1. **Baseline:** TF-IDF features + logistic regression (scikit-learn)
2. **Fine-tuned transformer:** DistilBERT trained with a hand-written PyTorch training loop
3. **Zero-shot foundation model:** BART-MNLI with no training at all

## Results

Test set: 3,096 reviews, 18% not recommended. Scores are for the *not recommended* class; thresholds were chosen on a separate validation set.

| Model | Precision | Recall | F1 | PR-AUC |
| --- | --- | --- | --- | --- |
| TF-IDF + logistic regression | 0.705 | 0.790 | 0.745 | 0.810 |
| **DistilBERT (fine-tuned)** | **0.710** | **0.834** | **0.767** | **0.824** |

On a 500-review sample, the zero-shot BART-MNLI model reached F1 0.680 with no labelled data (baseline 0.698, DistilBERT 0.756 on the same sample).

## Key findings

- **Fine-tuning helped, modestly.** DistilBERT gained about 2 F1 points, mostly through recall (0.790 → 0.834), catching around 25 more unhappy shoppers in the test set. The simple baseline gets most of the way there at a fraction of the cost.
- **Zero-shot is a strong starting point.** Without any training it came within 2 F1 points of the trained baseline: high precision (0.725) but lower recall (0.641).
- **Some "errors" are label noise.** The model's most confident mistakes were five-star rave reviews marked "not recommended", which are most likely mis-ticked boxes. Other hard cases: shoppers who complain but still recommend, and politely worded criticism.
- **Words the baseline learned** for unhappy shoppers include *disappointed*, *not flattering*, *unflattering*, *cheap*, *going back* and *wanted to (love)*.
- **Trend** has the highest share of unhappy shoppers, but it is the smallest department, so that figure is less reliable.

## Method

- **Data:** 22,641 women's clothing e-commerce reviews ([Hugging Face](https://huggingface.co/datasets/saattrupdan/womens-clothing-ecommerce-reviews)); 20,636 after removing empty and duplicate reviews.
- **Split:** stratified 70 / 15 / 15 train / validation / test. All tuning (regularisation strength, decision threshold, best epoch) used validation only; each model touched the test set once.
- **Leakage check:** the star rating predicts the label almost perfectly, so it was excluded; models learn from review text alone.
- **Class imbalance:** class-weighted losses, and metrics focused on the minority class (precision, recall, F1, PR-AUC) rather than accuracy.
- **Transformer training:** DistilBERT, max length 128, batch size 32, 2 epochs, AdamW (lr 2e-5) with linear warm-up, mixed precision on a T4 GPU, best epoch kept by validation PR-AUC.

## Run it yourself

Click **Open in Colab** above, then *Runtime → Change runtime type → T4 GPU* and *Runtime → Run all*. It takes about 10 minutes.

## What I'd do next

- Clean the noisy labels and re-measure
- Try a larger model (e.g. RoBERTa) and check whether the gain is worth the cost
- Calibrate probabilities so a score of 0.8 really means 80% likely
- Group flagged reviews by theme (sizing, fabric, colour) for actionable summaries
- Serve the best model through a small Flask API for a dashboard

## Tools

Python · pandas · NumPy · scikit-learn · PyTorch · Hugging Face Transformers and Datasets · Matplotlib · Google Colab
