#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import logging

def set_logger(name=None, verbose=False, handler=logging.StreamHandler()):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    #formatter = logging.Formatter('[%(levelname).1s%(name)s-%(asctime)s %(filename)s:%(funcName)s:%(lineno)3d] %(message)s', datefmt='%m-%d %H:%M:%S')
    formatter = logging.Formatter('[%(levelname).1s:%(name)s (%(asctime)s) %(funcName)s] %(message)s', datefmt='%m-%d %H:%M:%S')
    console_handler = handler
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


if __name__ == "__main__":
    message = "hello"
    logger = set_logger()
    logger.info(message)
    print("yes")