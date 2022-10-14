# Verb Cardinality Reduction for BioMedical Pred-Argument Graphs Extracted from Unstructured Text

![Maturity level-Prototype](https://img.shields.io/badge/Maturity%20Level-Prototype-red)
![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Introduction

Predicate-Argument Graphs extracted from the unstructured text have a high cardinality of verbs (Arguments), limiting the use of graphs. Particularly in the biomedical domain, there are no existing data sources that could use to train or map verbs. Reducing the verb count while not losing information is the key challenge.

`verbReduce` do not:

- Require existing resource for Biomedical domain
- Require 'Gold' verb set
- Require 'K' for verbs
- Require evaluation dataset


Given the unlabeled data, our approach provides a lookup table mapping source verb to target verb. 



## Architecture Diagram

![Untitled Diagram drawio (7)](https://user-images.githubusercontent.com/44647776/195863083-1c1c69a3-02e2-423d-b101-b2f68063d932.png)



## Setup

Run the following to setup the code

```
make install-dependencies
```

Tests:

```
pytest -s
```

## Running the code

We use the external libraries in the code, so it is useful to be familiar with the way these libraries work. The libraries are:

- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/starter/introduction.html)
  - In particular we use the [Lightning Module](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html) and [Lightning Datamodule](https://pytorch-lightning.readthedocs.io/en/stable/data/datamodule.html)
- [PySAT](https://pysathq.github.io/)
  - In particular we use [Minimum/minimal hitting set solver](https://pysathq.github.io/docs/html/api/examples/hitman.html)
- [Prefect](https://www.prefect.io/](https://docs.prefect.io/tutorials/first-steps/))
  - We only use the basic task-flow paradigm of prefect. 
- [Dynaconf](https://www.dynaconf.com/)
  - We use this to specify the parameters for the train and predict flow  

**Environment Variables**:

- `export PREFECT_HOME=<path where you have enough space>`
  -   Prefect stores output from task in local disk. Make sure to provide path where there is enough space.
- `export TOKENIZERS_PARALLELISM=false`
  - This is to disable the warning messages thrown by HuggingFace Tokenizers
- `export ENV_FOR_DYNACONF=default`
  -  This is to set the which environment variables you are going to run from the `settings.local.toml` (This file is not tracked by git and can vary with each local config). Please refer to this [link](https://www.dynaconf.com/settings_files/) for further information.


### Links
- [HOWTO: Training the Self-Supervised Model](https://github.com/AZ-AI/verbReduce/wiki/HOWTO:-Training-the-Self-Supervised-Model)
- [HOWTO: Generating Lookup Table
](https://github.com/AZ-AI/verbReduce/wiki/HOWTO:-Generating-Lookup-Table)



## Understanding the code


Addressing the challenge in three parts:

- [Identify potential set candidate substitute verbs](https://github.com/AZ-AI/verbReduce/wiki/Self-Supervised-Masked-Language-Model)
- [Reduce the cardinality of verbs](https://github.com/AZ-AI/verbReduce/wiki/Verb-Reduction-using-HittingSet-SAT-Solver)
- [Evaluate the accuracy of replacements](https://github.com/AZ-AI/verbReduce/wiki/Evaluate-the-accuracy-of-replacements)


## Features to be implemented
- [ ] Support multi-gpu training/inference (currently the code only supports one gpu)
- [ ] Use context in verb substitution prediction
- [ ] Deal with multiple token verbs. (currently the approach only uses found in vocabulary. If the verb is split into two tokens, we ignore it)

