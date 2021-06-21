# SpamSMSClassification
Classification on SMS Spam Collection Dataset using classical
machine learning approaches

## Dataset
[SMS Spam Collection v. 1](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/)

Download the archive by the [LINK](https://www.dt.fee.unicamp.br/~tiago/smsspamcollection/smsspamcollection.zip), extract its contents, and
provide the destination to the train.py via the ```--dataset``` key

### Dataset analysis
To run the analysis, use:

```python analyze_dataset.py --dataset <DATASET_PATH>```

## Approaches
- [ ] Logistic regression 

### Logistic regression
The baseline solution is the logistic regression one which is fast
to train.

To run the training, use the following command:

```python train.py --classifier logistic --dataset <DATASET_PATH>  --log <LOG_DIR>```

## Troubleshooting
### Stopwords
If you are using NLTK package for the first time, you might 
need to download stopwords by activating your
current venv and running 
```
import nltk
nltk.download('stopwords')
```