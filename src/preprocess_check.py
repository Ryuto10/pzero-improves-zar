# coding=utf-8

import json
from itertools import islice
from copy import deepcopy

from logzero import logger
from termcolor import colored

from preprocess import create_parser
from tokenizer import load_tokenizer
from utils import read_lines
from instances import GA, WO, NI, TARGET_CASES, EXO_INDEX


CASE_COLOR = {
    GA: 'cyan',
    WO: 'blue',
    NI: 'green',
    'pred': 'yellow',
    'mask': 'red',
    'ans': 'magenta'
}

EXO_NAMES = {value: key for key, value in EXO_INDEX.items()}
EXO_NAMES[-100] = '_'


def main():
    parser = create_parser()
    parser.add_argument('--num', type=int, default=10, help='The number of lines to display (default=10)')
    args = parser.parse_args()
    logger.info(args)

    logger.info(f'Type: {args.type}')
    tokenizer = load_tokenizer()

    for line in islice(read_lines(args.in_file), args.num):
        if line == '':
            print()
            continue
        instance = json.loads(line)

        if args.type == 'chunk':
            print(' '.join(''.join(surf for surf in chunk['surfs']) for chunk in instance))

        elif args.type == 'cloze':
            print(tokenizer.decode(instance['input_ids']))

        elif args.type == 'pzero':
            input_tokens = tokenizer.convert_ids_to_tokens(instance['input_ids'])
            idx = instance['masked_idx']
            input_tokens[idx] = colored(input_tokens[idx], CASE_COLOR['mask'])
            for idx in instance['gold_ids']:
                input_tokens[idx] = colored(input_tokens[idx], CASE_COLOR['ans'])
            print(''.join(input_tokens))

        elif args.type == 'ntc':
            for pas in instance['pas_list']:
                sents = deepcopy(instance['sents'])
                sent_idx = pas['prd_sent_idx']
                for word_idx in pas['prd_word_ids']:
                    sents[sent_idx][word_idx] = colored(sents[sent_idx][word_idx], CASE_COLOR['pred'])

                for case_name, gold_label in pas['gold_labels'].items():
                    for gold_case in gold_label['gold_cases']:
                        sent_idx = gold_case['sent_idx']
                        word_idx = gold_case['word_idx']
                        sents[sent_idx][word_idx] = colored(sents[sent_idx][word_idx], CASE_COLOR[case_name])

                for sent in sents:
                    print(''.join(sent))
                print()

        elif args.type == 'as':
            input_tokens = instance['input_tokens']
            for idx in instance['predicate_position_ids']:
                input_tokens[idx] = colored(input_tokens[idx], CASE_COLOR['pred'])
            for case_name, gold_ids in instance['gold_positions'].items():
                for idx in gold_ids:
                    input_tokens[idx] = colored(input_tokens[idx], CASE_COLOR[case_name])
            print(''.join(input_tokens))
            exo = ', '.join(f"{case_name}: {EXO_NAMES[instance['exo_idx'][case_name]]}" for case_name in TARGET_CASES)
            print(f' - [exo] {exo}')

        elif args.type == 'as-pzero':
            case_name = instance['case_name']
            input_tokens = instance['input_tokens']
            idx = instance['mask_position_id']
            input_tokens[idx] = colored(input_tokens[idx], CASE_COLOR['mask'])
            for idx in instance['predicate_position_ids']:
                input_tokens[idx] = colored(input_tokens[idx], CASE_COLOR['pred'])
            for idx in instance['gold_positions']:
                input_tokens[idx] = colored(input_tokens[idx], CASE_COLOR[case_name])
            print(''.join(input_tokens))
            print(f" - [exo] {EXO_NAMES[instance['exo_idx']]}")

        else:
            raise ValueError(f'Unsupported value: {args.type}')


if __name__ == '__main__':
    main()
