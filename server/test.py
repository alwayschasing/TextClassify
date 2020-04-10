#!/usr/bin/env python
# -*- encoding:utf-8 -*-

from multiprocessing import Process, Event, Queue
import time
import requests
import json

class TestServer(Process):
    def __init__(self):
        #super(Process, self).__init__()
        Process.__init__(self)

    def run(self):
        print("work")
        time.sleep(3)


def tts_req():
    st = time.time()
    url = "http://127.0.0.1:9001/tts-classify" 
    data = {
        "texts":["玩玉，最关注的四个问题 现如今玩玉赏玉已经成为了一部分人的终生爱好，对于刚踏进门的初学者来说，对玩玉还存在很多疑问。大概归纳起来，下面四个问题是提及率最高的：价格、产地、籽料、真假。今天，小编和广大玉石爱好者们就这四个问题进行简单的探讨。 一、关于价格很多人反"]
    } 
    data = json.dumps(data)
    headers = {"Content-Type":"application/json"}
    res = requests.post(url,data=data,headers=headers)
    print(res)
    print(res.text)
    ed = time.time()
    cost = (ed - st)*1000
    print("cost:%d ms\n"%(cost))

if __name__ == "__main__":
    tts_req()
    #main()
    #test_server = TestServer()
    #test_server.start()
    #test_server.join()
