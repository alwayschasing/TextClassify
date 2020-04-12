#!/usr/bin/env python
# -*- encoding:utf-8 -*-
import logging
import os
import uuid
import zmq

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

def auto_bind(socket):
    try:
        tmp_dir = os.environ['ZEROMQ_SOCK_TMP_DIR']
        if not os.path.exists(tmp_dir):
            raise ValueError('This directory for sockets ({}) does not seems to exist.'.format(tmp_dir))
        tmp_dir = os.path.join(tmp_dir, str(uuid.uuid1())[:8])
    except KeyError:
        tmp_dir = '*'
    
    socket.bind('ipc://{}'.format(tmp_dir))
    return socket.getsocketopt(zmq.LAST_ENDPOINT).decode('ascii')

if __name__ == "__main__":
    message = "hello"
    logger = set_logger()
    logger.info(message)
    print("yes")