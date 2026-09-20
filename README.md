# Latin Poetry Meter Classification

Classifiers that try to predict the **metrical scheme of a line of Latin poetry** from its text, trained on the expert-scanned [Pedecerto](http://www.pedecerto.eu) corpus.

**Headline finding:** with a leak-free evaluation, bag-of-words models (Naive Bayes, Random Forest, LightGBM) barely beat a "always guess hexameter" baseline. The high scores the original notebook protocol produces come from train/test leakage. Details and numbers are below.

## Task

Given a line of Latin verse (lowercased, punctuation stripped), predict its **meter label**, for example `H` (dactylic hexameter) or `P` (pentameter) in Pedecerto's coding.

## Data

The corpus is **Pedecerto. Digital Latin Metre**, produced at the Università degli Studi di Udine and Università Ca' Foscari Venezia (Emanuela Colombi, Luca Mondin, Luigi Tessarolo, Andrea Bacianini). Project site: <http://www.pedecerto.eu>.

The files in [`ScannedPoetry/`](ScannedPoetry) are Pedecerto's XML files, one per text, redistributed unmodified. They are licensed **[CC BY-NC-ND 4.0](http://creativecommons.org/licenses/by-nc-nd/4.0/legalcode)**, so attribution is required, use is non-commercial, and the files may not be altered. Please respect that license and cite Pedecerto if you reuse them.

Each `<line>` carries a line number (`name`), a meter label (`meter`) and a foot pattern (`pattern`, for example `DDSS`), and contains `<word>` elements with syllable and word-boundary annotations.

**Corpus statistics:**

| | |
|---|---|
| Texts (XML files) | 86, including Virgil, Ovid, Horace, Lucan, Statius, Seneca, Martial, Persius, Catullus and others |
| Lines | 132,702 (~828,000 word tokens) |
| Usable lines (non-empty text, scanned, has a meter label) | **130,320** |
| Distinct meter labels | **21** (the 11 lines with an empty label are dropped) |
| Most common label | `H` (hexameter), **84.5%** of usable lines |
| Rarest labels | `D3` (92 lines), `A` (63), `S` (33), `C` (10), `X` (8), `L` (7) |

The classes are extremely imbalanced, so plain accuracy is a poor metric. **Always guessing hexameter scores 84.5% accuracy** on the full corpus, with a macro-F1 of only 0.04.

## Method

The original exploratory notebook is [`PoetryClass.ipynb`](PoetryClass.ipynb) (Multinomial Naive Bayes, Random Forest, LightGBM and multilingual BERT on bag-of-words / WordPiece features). [`latin_rerun/run_experiments.py`](latin_rerun/run_experiments.py) re-runs the three classical models with a corrected protocol:

- **Full corpus.** Lines are selected with `.//body//line`, not `.//body/line`. The notebook's XPath matches only lines that are *direct children* of `<body>`, so it reads about 14,400 lines from 30 of the 86 texts (15 classes) and misses about 89% of the usable data.
- **Split first, balance second.** The notebook upsamples rare classes with `resample(..., replace=True)` *before* the train/test split, which puts copies of the same line in both sets. The rerun splits first, learns the vocabulary on the training data only, and does no row duplication.
- **Two evaluation regimes:** a stratified random 80/20 line split (26,064 test lines), and **5-fold cross-validation that holds out whole texts**, so no work appears in both train and test.
- **Two class-handling variants:** *unweighted* (standard models) and *class-balanced* (uniform prior for Naive Bayes, `balanced` class weights for Random Forest and LightGBM).
- **Metrics:** accuracy, macro-F1 and balanced accuracy, always shown next to a majority-class baseline.
- Changes from the notebook: `CountVectorizer(min_df=2)`, and class weights instead of oversampling. Models keep the notebook's settings (Random Forest: 100 trees, max depth 100; LightGBM: 100 rounds, learning rate 0.1, 31 leaves). Seed is 42.

**Multilingual BERT was not rerun**, because it needs a GPU. The notebook's BERT split has the same leakage problem (it also upsamples before splitting), so any BERT accuracy from the notebook should not be trusted.

## Results

### 1. How much the original protocol was inflated

Same 14,396-line subset the notebook actually uses:

| | Accuracy | Macro-F1 |
|---|---|---|
| Notebook protocol, Naive Bayes | **95.2%** | 0.950 |
| Notebook protocol, Random Forest | **90.1%** | 0.909 |
| Same data, split first, Naive Bayes (class-balanced) | 71.1% | 0.128 |
| Same data, split first, Random Forest (class-balanced) | 59.7% | 0.147 |
| Always guess hexameter | 74.5% | 0.057 |

In the notebook protocol, **93.6% of test lines have an exact duplicate in the training set**, which is why the scores look so good.

### 2. Full corpus, leak-free

Random 80/20 line split (26,064 test lines):

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always guess hexameter | 84.5% | 0.044 |
| Naive Bayes (unweighted) | **86.5%** | 0.074 |
| Random Forest (unweighted) | 84.6% | 0.049 |
| LightGBM (unweighted) | 84.4% | 0.044 |
| Naive Bayes (class-balanced) | 77.6% | 0.094 |
| Random Forest (class-balanced) | 58.2% | 0.115 |
| LightGBM (class-balanced) | 35.4% | 0.060 |

Held-out texts, 5-fold mean ± std:

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Always guess hexameter | 84.5 ± 3.9% | 0.081 ± 0.031 |
| Naive Bayes (unweighted) | 85.8 ± 3.3% | 0.123 ± 0.048 |
| Random Forest (unweighted) | 84.6 ± 3.9% | 0.082 ± 0.031 |
| LightGBM (unweighted) | 76.1 ± 8.9% | 0.085 ± 0.030 |
| Naive Bayes (class-balanced) | 74.6 ± 2.4% | 0.130 ± 0.041 |
| Random Forest (class-balanced) | 56.9 ± 2.5% | 0.170 ± 0.082 |
| LightGBM (class-balanced) | 32.8 ± 1.6% | 0.087 ± 0.024 |

Macro-F1 is computed over the labels present in each test set. Raw per-fold numbers are in [`results_unweighted.json`](latin_rerun/results_unweighted.json) and [`results_class_balanced.json`](latin_rerun/results_class_balanced.json).

### Interpretation

- The best accuracy anywhere (Naive Bayes, 86.5%) is about **2 points above the majority baseline**, and the tree models are at or below it. Balancing the classes raises macro-F1 (best 0.17) but wrecks accuracy, and the balanced LightGBM is unstable.
- Holding out whole texts changes little for these models, which suggests they are not simply memorizing individual works, but rather that word counts carry very little metrical signal.
- This is expected: meter is determined by syllable quantity and line structure, which a bag of words does not encode.

## Running it

```bash
pip install numpy scikit-learn lightgbm
python latin_rerun/run_experiments.py ScannedPoetry results.json orig,fixed         # class-balanced variant
UNWEIGHTED=1 python latin_rerun/run_experiments.py ScannedPoetry results.json fixed   # unweighted variant
```

The script uses only the standard library for XML parsing. The whole run takes a few minutes on a laptop CPU.

The notebook was written for Python 3.10; edit `xml_folder` in its first code cell to point at `ScannedPoetry/`. BERT fine-tuning needs a GPU.

## Known issues and next steps

- **The notebook itself is unchanged** and still has the partial-corpus parser and the oversample-before-split leakage. Use `latin_rerun/run_experiments.py` for trustworthy numbers, or port its protocol into the notebook.
- **Bag-of-words is the wrong representation for this task.** More promising features should be derived from the *text itself*: estimated syllable counts (vowel groups), word-length patterns, and character n-grams, which capture syllable structure. Do **not** use the XML's `sy`, `wb` or `pattern` attributes as features. Those are the scansion annotations, so using them would leak the label.
- A sequence model over syllable-level features, or BERT with a leak-free split, has not been evaluated here.

## Acknowledgements

Corpus: Pedecerto. Digital Latin Metre (Università degli Studi di Udine; Università Ca' Foscari Venezia), <http://www.pedecerto.eu>, CC BY-NC-ND 4.0.
