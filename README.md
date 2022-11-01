# Verb Cardinality Reduction for BioMedical Pred-Argument Graphs Extracted from Unstructured Text via Self-Supervised Language Models


## Introduction


Predictate-Argument Graphs extracted from unstructured text, has a high cardinality of verbs (Arguments) resulting in limiting the use of graphs. Particularly in the biomedical domain, there are no existing data sources that could use to train or map verbs. The challenge of reducing the verb count while not losing information is key challenge.

`verbReduce` do not:

- Require existing resource for Biomedical domain
- Require 'Gold' verb set
- Require 'K' for verbs
- Require evaluation dataset

## Setup

Run the following to setup the code

```
make install-dependencies
```

Tests:

```
pytest -s
```

## Architecture Diagram


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
- [HOWTO: Training the Self-Supervised Model](https://github.com/AstraZeneca/verbReduce/wiki/HOWTO:-Training-the-Self-Supervised-Model)
- [HOWTO: Generating Lookup Table
](https://github.com/AstraZeneca/verbReduce/wiki/HOWTO:-Generating-Lookup-Table)



## Understanding the code


Addressing the challenge in three parts:

- [Identify potential set candidate substitute verbs](https://github.com/AstraZeneca/verbReduce/wiki/Self-Supervised-Masked-Language-Model)
- [Reduce the cardinality of verbs](https://github.com/AstraZeneca/verbReduce/wiki/Verb-Reduction)
- [Evaluate the accuracy of replacements](https://github.com/AstraZeneca/verbReduce/wiki/Evaluate-the-accuracy-of-replacements)


## Features to be implemented
- [ ] Support multi-gpu training/inference (currently the code only supports one gpu)
- [ ] Use context in verb substitution prediction
- [ ] Deal with multiple token verbs. (currently the approach only uses found in vocabulary. If the verb is split into two tokens, we ignore it)

