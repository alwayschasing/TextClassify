#!/usr/bin/env python
# -*-utf-8-*-


import jieba
import json
import logging
logging.basicConfig(level=logging.DEBUG,format='[%(asctime)s-%(levelname)%s] %(message)s',datefmt='%Y-%m-%d %H:%M:%S')


def load_forward_data(input_file, delimiter='\t', target_index=2):
    text_list = []
    with open(input_file,"r",encoding='utf-8') as fp:
        n_error = 0
        for line in fp:
            items = line.strip().split('\t')
            if len(items) < target_index + 1:
                continue

            try:
                json_text = json.loads(items[target_index], encoding='utf-8')
            except ValueError:
                logging.debug("[load error]%s"%(items[target_index]))
                n_error += 1
                continue

            title = json_text["title"]
            content = json_text["content"]
            text_list.append(title + ' ' + content)

    logging.info('[finish load text] %d lines'%(len(text_list)))
    return text_list


def load_stop_words(stop_file,encoding='utf-8'):
    words = []
    with open(stop_file,"r",encoding=encoding) as fp:
        for line in fp:
            words.append(line.strip())
    return words


def build_words_dict(forward_file, words_file, config):
    text_list = load_forward_data(forward_file)
    stop_words = load_stop_words(config["stop_file"])














