#! /bin/bash

INPUT=${1}

JUMAN_DIC="/usr/local/lib/mecab/dic/jumandic"
JUMAN_DEP_MODEL="/usr/local/lib/cabocha/model/dep.juman.model"
JUMAN_CHUNK_MODEL="/usr/local/lib/cabocha/model/chunk.juman.model"
JUMAN_NE_MODEL="/usr/local/lib/cabocha/model/ne.juman.model"

CABOCHA_CMD="/usr/local/bin/cabocha -f1 -n1 -P JUMAN -d ${JUMAN_DIC} -m ${JUMAN_DEP_MODEL} -M ${JUMAN_CHUNK_MODEL} -N ${JUMAN_NE_MODEL}"

${CABOCHA_CMD} < ${INPUT}
