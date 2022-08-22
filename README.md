# Is a Question Decomposition Unit All We Need?

This repository contains the data and the code used to obtain the results in [Patel et al.](https://arxiv.org/pdf/2205.12538.pdf)

Installation:

```
pip install -r src/requirements.txt
```

Running the code:

1) For results with GPT3:

```
python src/GPT3.py --dataset_name <dataset_name> --API_TOKEN <OpenAI API TOKEN>
```

dataset_name values:
hotpot
drop
qasc
multirc
strategyqa
svamp
mathqa
break

2) For results with RoBERTa:

```
python src/roberta.py --dataset_name <dataset_name>
```

dataset_name values:
hotpot
drop
break
multirc
svamp

3) For results with RoBERTa for strategyqa:

```
python src/strategy_roberta.py --dataset_name strategyqa
```

Output of each of these are three files:

one, ending with _trial, is the same as src/dataset_name.json, along with generated responses.

second, ending with _original.json, is a json file consisting of ground truth answers and generated answers for original (undecomposed) questions.

third, ending with _decomposition.json, is a json consisting of ground truth answers and generated answers for decomposed questions.

Evalutations:

```
python src/evalute.py --dataset_file <original/decomposition.json>
```
