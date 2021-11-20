# Training

## Pretraining
- Cloze task
    ```bash
    $ poetry run python src/training.py --yaml_file src/args/pretraining_cloze_params.yml
    ```
- Pzero task
    ```bash
    $ poetry run python src/training.py --yaml_file src/args/pretraining_pzero_params.yml
    ```
### How to write a yaml file of pretraining
- please see [the code](../src/training.py#L59-L99)

## Fine-tuning
- AS model
    ```bash
    $ poetry run python src/training.py --yaml_file src/args/finetuning_as_params.yml
    ```
  
- AS-Pzero model
    ```bash
    $ poetry run python src/training.py --yaml_file src/args/finetuning_as_pzero_params.yml
    ```
### How to write a yaml file of fine-tuning
- please see [the code](../src/training.py#L183-L219)
