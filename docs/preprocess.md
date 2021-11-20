# Preprocessing

## For Pretraining
- First, we convert the raw text to full-width:
  ```bash
  $ poetry run python src/preprocess.py \
    --in [path to raw text] \
    --type "raw" \
    > [path to output (full-with text)]
  ```
  - Example of `full-width text`: `tests/samples/raw.txt`


- Next, we parse the text with Japanese Dependency Parser *CaboCha* (Kudo and Matsumoto, 2002):
  ```bash
  $ poetry run bash src/preprocess_with_cabocha.sh \
    [path to full-width text] \
    > [path to output (text parsed with CaboCha)]
  ```
  - Example of `full-width text`: `tests/samples/raw.txt`
  - Example of `text parsed with CaboCha`: `tests/samples/raw.parsed.txt`


### Cloze Task
- Create instances for training the model in Cloze Task
  ```bash
  $ poetry run python src/preprocess.py \
    --in [path to text parsed with CaboCha] \
    --type "cloze" \
    > [path to output]
  ```
  - Example of `text parsed with CaboCha`: `tests/samples/raw.parsed.txt`
  - Example of `output`: `tests/samples/cloze.instances.jsonl`


### Pzero Task
- Create instances for training the model in Pzero Task
  ```bash
  $ poetry run python src/preprocess.py \
    --in [path to text parsed with CaboCha] \
    --type "pzero" \
    > [path to output]
  ```
  - Example of `text parsed with CaboCha`: `tests/samples/raw.parsed.txt`
  - Example of `output`: `tests/samples/pzero.instances.jsonl`


## For Finetuning

### AS Model
- Create instances for training the AS model
  ```bash
  $ poetry run python src/preprocess.py \
    --in [path to directory of NAIST Text Corpus] \
    --type "as" \
    > [path to output]
  ```
  - Example of `directory of NAIST Text Corpus`: `tests/samples/dummy_ntc`
  - Example of `output`: `tests/samples/as.instances.jsonl`


### AS-Pzero Model
- Create instances for training the AS-Pzero model
  ```bash
  $ poetry run python src/preprocess.py \
    --in [path to directory of NAIST Text Corpus] \
    --type "as-pzero" \
    > [path to output]
  ```
  - Example of `directory of NAIST Text Corpus`: `tests/samples/dummy_ntc`
  - Example of `output`: `tests/samples/as-pzero.instances.jsonl`


## Visualize the created instances
```bash
$ poetry run python src/preprocess_check.py \
  --in [path to created instance] \
  --type [choose from 'cloze', 'pzero', 'as', 'as-pzero'] \
  --num [The number of instances to print]
```
