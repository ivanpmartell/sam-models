# Single amino-acid mutation Models (SAM-Models) research

A set of python scripts to train, test, and validate machine learning models for missense mutation refinement of secondary structure.

The data used for the training of the models can be generated from [PDB-SAM](https://github.com/ivanpmartell/pdb-sam).

## Required Python libraries
```
torch
lightning
biopython
scikit-learn
numpy
pandas
matplotlib
argparse
```

 If utilizing Ubuntu 18.04 or 20.04, we have provided a script `setup.sh` that will automate the acquisition of the required software and libraries.

## Protein Structure Prediction Methods (Predictors)

They can be obtained from the following locations:

### Secondary Structure Prediction Methods
- SSPro8 from [here](https://scratch.proteomics.ics.uci.edu/explanation.html#SSpro8)
- RaptorX PropertyPredict from [here](https://github.com/realbigws/Predict_Property)
- SPOT1D from [here](https://zhouyq-lab.szbl.ac.cn/download/)
- SPOT1D-Single from [here](https://github.com/jas-preet/SPOT-1D-Single)
- SPOT1D-LM from [here](https://github.com/jas-preet/SPOT-1D-LM)

### Tertiary Structure Prediction Methods
- Alphafold 2 from [here](https://github.com/google-deepmind/alphafold)
- ESMFold from [here](https://github.com/facebookresearch/esm)
- ColabFold from [here](https://github.com/YoshitakaMo/localcolabfold)
- RGN2 from [here](https://github.com/aqlaboratory/rgn2) and made into a locally runable program through `extra/rgn2_local_files` in [PDB-SAM](https://github.com/ivanpmartell/pdb-sam).

The setup of these methods are hardware-dependent and as such must be manually installed by following each of their installation procedures.

## Dataset creation and prediction acquisition.

Same procedure as [PDB-SAM](https://github.com/ivanpmartell/pdb-sam).
This dataset can then be converted into a numpy pickle format to use during training and testing procedures.

## Training, Testing, Predicting

Scripts for training, testing, and predicting using tree-type and neural-type models are given in their respective folders inside the `models` directory.

For example, to train a neural-type model use the `models/train/nn.py` script.

Likewise, to test a tree-type model use the `models/test/trees.py` script.

To make predictions, it is necessary to already have the outputs from predictors of choice and a trained model. Predictions can be done for tree-type, neural-type or a majority-based model. Unlike the others, the majority-based model takes the predictors outputs and returns the most predicted structure for a given amino-acid,  thus no actual training is necessary.

## Evaluation

Once the output from predictors have been obtained, cross validation can be performed following the cross-validation script `cross_validation.sh` that uses the following python script:

```
models/cross_validation.py
```

Likewise, feature selection to obtain the importance of the predictors can be done through the `feature_importance.sh` script , which uses the following python script:

```
models/features.py
```

## Model knowledge interpretation

The knowledge can be extracted through three scripts

- `models/visualize_trees.py`: Script to visualize all decision from tree-type models as an image or a text file.

- `models/tree_rule_analysis.py`: Utilizes the text visualization to create a confusion matrix of the secondary structure classes that the tree can predict.

- `models/tree_decisions.py`: Creates a list of human-readable decisions that the tree takes to predict a certain secondary structure class.