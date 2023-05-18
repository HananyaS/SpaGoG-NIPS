# SpaGoG: Graph of Graphs to classify tabular data with large fractions of missing data

This repository is the official implementation of **"SpaGoG: Graph of Graphs to classify tabular data with large fractions of missing data"**. The project was implementes in Python 3.8. 

[!SpaGoG scheme figure](/figures/spagog_scheme_figure.pdf)
<image src="/figures/spagog_scheme_figure.pdf/">

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install the requirements for this implementation, please use the following command:

For pip:

```setup
python -m venv <env_name>
source <env_name>/bin/activate
pip install -r requirements.txt
```

For conda:

```setup
conda env create -f environment.yml -n <env_name>
conda activate <env_name>
```


## Training & Evaluation

To reproduce the main experimental results from the paper (See tables [1]-[3]), run the following command:

```train_eval
python main.py --dataset <dataset> --model <model> --kfolds <kfolds>
```


Args:

* **dataset**: The name of the dataset to run the models on. Options: ["Spam", "Iris", "Banknote", "Accent", "Sensorless", "Wine", "Credit", "Wifi", "Ecoli", "Sonar", "Breast", "HTRU2", "Parkinson", "CiteSeer", "PubMed", "Cora"] (case-sensitive).
* **model**: The SpaGoG model to run. Additionally, we add the GC model, for reproducing the results appear in Supp. Mat. Table S4. Options: ["gnc", "gc+nc", "gc"].
* **kfolds**: Number of folds to preform k-folds cross validation (in the paper, **kfolds** was set to 10). Setting **kfolds** to None, ignores the k-folds cross-validation process, and runs the model only once, with predfined splits. Options: [None, int > 1], Default: 10.
* **verbosity**: The verbosity level of the printed outputs. Options: [0, 1, 2], Default: 1. 

## Results

Our model has tested on various datasets with various missing rate. Here are the results:

| Dataset    | GNC          | GC+NC        |
| ---------- | ------------ | ------------ |
| Spam       | .959 ± .005  | .965 ± .005  |
| Credit     | .755 ± .008  | .759 ± .006  |
| Breast     | .98 +- .011  | .986 ± .01   |
| Parkinson  | .785 +- .027 | .781 ± .051  |
| Ecoli      | .588 ± .04   | .571 ± .05   |
| Accent     | .486 ± .038  | .489 ± .055  |
| Iris       | .763 ± .08   | .74 ± .122   |
| Sonar      | .719 ± .047  | .705 ± .040  |
| Wifi       | .819 +- .008 | .833 ± .007  |
| Wine White | .472 ± .018  | .482 ± .011  |
| Banknote   | .787 +- .013 | .776 ± .022  |
| HTRU2      | .965 ± .002  | .9642 ± .003 |
| Sensorless | .776 +- .008 | .730 ± .008  |
| Cora       | .676 +- .016 | .770 ± .013  |
| CiteSeer   | .509 +- .016 | .512 ± .141  |
| PubMed     | .801 +- .004 | .828 ± .003  |

where **Spam, Credit, Breast** tested with 20% missing rate, **Parkinson, Ecoli, Accent, Iris, Sonar, Wifi, Wine White, Banknote, HTRU2, Sensorless** with 50% missing rate and **Cora, CiteSeer, PubMed** tested with 99% misising rate.
