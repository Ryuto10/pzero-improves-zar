# Pseudo Zero Pronoun Resolution Improves Zero Anaphora Resolution
This is the official repository of following paper:
- Title: Pseudo Zero Pronoun Resolution Improves Zero Anaphora Resolution
- Author: Ryuto Konno, Shun Kiyono, Yuichiroh Matsubayashi, Hiroki Ouchi, Kentaro Inui
- Conference: The 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP 2021)
- Conference paper: https://aclanthology.org/2021.emnlp-main.308/
- arXiv version: https://arxiv.org/abs/2104.07425v2

## Requirements
- python 3.8
- poetry 1.1.10
- [docker](https://www.docker.com/) 20.10.7 or higher
- docker-compose 1.21.0 or higher

## Setup with Docker
- Build
    ```bash
    $ git clone https://github.com/Ryuto10/pzero-improves-zar
    $ cd pzero-improves-zar
    $ docker-compose -f docker/docker-compose.yml up -d --build
    ```
- Enter the docker container (Interactive)
    ```bash
    $ docker exec -it pzero-improves-zar bash
    $ poetry install
    ```

## Run
- [Using ZAR Model](docs/demo.md)
- [Preprocessing the corpus](docs/preprocess.md)
- [Training](docs/training.md)
- [Decoding](docs/decoding.md)
- [Evaluation](docs/evaluation.md)

## Citation
```
@InProceedings{konno-etal-2021-pseudo,
    author    = {Ryuto Konno and Shun Kiyono and Yuichiroh Matsubayashi and Hiroki Ouchi and Kentaro Inui},
    title     = {{Pseudo Zero Pronoun Resolution Improves Zero Anaphora Resolution}},
    booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing}
    month     = nov,
    year      = "2021",
    publisher = {Association for Computational Linguistics},
    url       = {https://aclanthology.org/2021.emnlp-main.308},
    pages     = "3790--3806",
}
```
