# Decoding

- Run
    ```bash
    $ poetry run python src/decode.py \
      --data [path to preprocessed file] \
      --yaml_file [path to yaml file] 
    ```

- Example of results
    ```json lines
    {"file": "path-to-file", "pred": {"sent": 0, "id": 13}, "ga": {"sent": 0, "id": 5}, "o": {"sent": 0, "id": 11}}
    {"file": "path-to-file", "pred": {"sent": 0, "id": 15}}
    {"file": "path-to-file", "pred": {"sent": 0, "id": 3}, "ga": {"sent": 0, "id": 5}, "o": {"sent": 0, "id": 1}}
    {"file": "path-to-file", "pred": {"sent": 1, "id": 11}, "ga": {"sent": 1, "id": 0}, "o": {"sent": 1, "id": 7}, "ni": {"sent": 1, "id": 9}}
    {"file": "path-to-file", "pred": {"sent": 1, "id": 17}, "ga": {"sent": 1, "id": 0}, "o": {"sent": 1, "id": 14}}
    ```
