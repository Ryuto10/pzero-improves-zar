# Zero Anaphora Resolution of Raw Text

### 1. Prepare raw text and pre-processing
- Preparing the raw text to resolve zero anaphor (`sample.txt`)
    ```
    デフレ色を深める日本経済はいよいよ容易ならざる局面を迎えたと思われる。
    政府は今年４月、所得税を半分にするという政策を表明した。
    ```

- Tokenizing the raw text using Japanese dependency parser *Cabocha*
    - ※ It is assumed that you are in a docker container (please see [README.md](../readme.md#setup-with-docker))
    ```bash
    $ bash src/preprocess_with_cabocha.sh samples/raw.txt > samples/raw.parsed.txt
    ```

- Setting the positions of target predicates
    - In the parsed text, add a number at the end of the line corresponding to the target predicate. The added numbers are ...
        - separated by a tab
        - natural numbers greater than 0
        - different between different predicates
        - the same when the tokens refer to the samce predicate
    - For example:（`raw_text.parsed.txt`）
      - '1' is assigned to the target predicate '深める'
      - '2' is assigned to the target predicate '迎えた'
      - '5' is assigned to the target predicate '表明した' (multiple tokens)
    ```
    * 0 1D 1/2 1.950401
    デフレ  名詞,普通名詞,*,*,デフレ,でふれ,代表表記:デフレ/でふれ カテゴリ:抽象物 ドメイン:ビジネス;政治   O
    色      名詞,普通名詞,*,*,色,いろ,代表表記:色/いろ 漢字読み:訓 カテゴリ:色      O
    を      助詞,格助詞,*,*,を,を,* O
    * 1 2D 0/0 1.349548
    深める  動詞,*,母音動詞,基本形,深める,ふかめる,代表表記:深める/ふかめる 自他動詞:自:深まる/ふかまる 形容詞派生:深い/ふかい      O       1
    * 2 6D 1/2 1.333196
    日本    名詞,地名,*,*,日本,にほん,代表表記:日本/にほん 地名:国  B-LOCATION
    経済    名詞,普通名詞,*,*,経済,けいざい,代表表記:経済/けいざい カテゴリ:抽象物 ドメイン:ビジネス;政治   O
    は      助詞,副助詞,*,*,は,は,* O
    .
    .
    .
    * 5 6D 0/1 2.367888
    局面	名詞,普通名詞,*,*,局面,きょくめん,代表表記:局面/きょくめん カテゴリ:抽象物	O
    を	助詞,格助詞,*,*,を,を,*	O
    * 6 7D 0/1 1.333196
    迎えた	動詞,*,母音動詞,タ形,迎える,むかえた,代表表記:迎える/むかえる 反義:動詞:送る/おくる	O	2
    と	助詞,格助詞,*,*,と,と,*	O
    .
    .
    .
    * 7 8D 0/1 1.677621
    政策    名詞,普通名詞,*,*,政策,せいさく,代表表記:政策/せいさく カテゴリ:抽象物 ドメイン:政治    O
    を      助詞,格助詞,*,*,を,を,* O
    * 8 -1D 1/1 0.000000
    表明    名詞,サ変名詞,*,*,表明,ひょうめい,代表表記:表明/ひょうめい 補文ト カテゴリ:抽象物       O       5
    した    動詞,*,サ変動詞,タ形,する,した,連語     O       5
    。      特殊,句点,*,*,。,。,*   O
    EOS
    ```

### 2. Download the model parameters
- You can download the model parameters from the [Github releases](https://github.com/Ryuto10/pzero-improves-zar/releases).
- After downloading, move the model parameters from the download directory to the data directory.

### 3. Prediction of the model for zero anaphora resolution
- Model Path
    - Base Model: `data/cloze.as.seed5000.model`
    - Proposed Model: `data/pzero.as-pzero.seed5000.model`
- Run
    - ※ Here please change 'model_type' depending on the model you want to use.
    ```bash
    $ INPUT_TEXT=/path/to/parsed_text
    $ MODEL=/path/to/model
    $ poetry run python src/run.py --parsed_text $INPUT_TEXT --model $MODEL --model_type "as-pzero"
    ```
- Results (Target predicates are '深める', '迎えた', and '表明した')
    ```
    デフレ色を深める日本経済はいよいよ容易ならざる局面を迎えたと思われる。
    政府は今年４月、所得税を半分にするという政策を表明した。

    pred: 深める
    ga: 経済
    o: 色

    pred: 迎えた
    ga: 経済
    o: 局面

    pred: した
    ga: 政府
    o: 政策
    ```
