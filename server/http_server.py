#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import sys
sys.path.append('../')
from multiprocessing import Process, Event, Queue
import time
import threading
import logging
import uuid
from run_predict import *
from helper import set_logger

class Test(Process):
    def __init__(self):
        super(Process, self).__init__()

    def run(self):
        print("work")
        time.sleep(3)


class HTTPServer(Process):
    def __init__(self, config, read_to_classify_que, classify_res_que, num_worker=1, logger=logging.getLogger("sver")):
        #super(Process,self).__init__()
        Process.__init__(self)
        self.config = config
        self.is_ready = Event()
        self.classify_res = dict()
        self.read_to_classify_que = read_to_classify_que
        self.classify_res_que = classify_res_que
        self.num_worker = num_worker
        self.logger=logger
        logger.info("finish init HTTPServer")
    
    def create_flask_app(self):
        try:
            from flask import Flask, request
            from flask_compress import Compress
            # from flask_cors import CORS
            from flask_json import FlaskJSON, as_json, JsonError
        except ImportError:
            raise ImportError()

        app = Flask(__name__)
        #app.config['SWAGGER'] = {
        #    'title':'Colors API',
        #    'uiversion':3,
        #    "openapi":"3.0.2"
        #}
        self.create_classify_worker(self.num_worker, self.read_to_classify_que,self.classify_res_que, self.config)

        @app.route('/tts-classify', methods=['POST'])
        @as_json
        def tts_classify():
            req_data = request.form if request.form else request.json
            req_data['req_id'] = uuid.uuid1()
            req_id = req_data['req_id']
            texts = req_data['texts'] # text list
            req_num = len(texts)
            req_time = time.time()
            req_data['req_time'] = req_time
            self.logger.debug("request %s"%(texts[0]))
            self.read_to_classify_que.put(req_data)
            while req_id not in self.classify_res:
                try:
                    res_item = self.classify_res_que.get()
                except:
                    continue
                if res_item is not None:
                    self.classify_res["req_id"] = res_item["req_id"]
            
            if req_id in self.classify_res:
                return classify_res[req_id]
            else:
                return None

        # CORS(app, origins=self.args.cors)
        FlaskJSON(app)
        Compress().init_app(app)
        return app

    def receiver(self):
        while True:
            data_item = self.classify_res_que.get()
            self.classify_res[data_item['req_id']] = data_item

    def run(self):
        # 启动分类结果接收线程
        self.logger.info("start run")
        receive_thread = threading.Thread(target=self.receiver)
        receive_thread.start()

        self.logger.info("start create app")
        app = self.create_flask_app()
        self.is_ready.set()
        app.run(port=self.config["http_port"], threaded=True, host='0.0.0.0')
        self.logger.info("list to port:%d"%(self.config["http_port"]))
        receive_thread.join()

    def create_classify_worker(self, num_worker, read_to_classify_que, classify_res_que, config):
        for i in range(num_worker):
            tts_server = TtsClassifyWorker(read_to_classify_que, classify_res_que, config, workerid=i, logger=self.logger)
            tts_server.start()
        

class TtsClassifyWorker(Process):
    def __init__(self, ready_to_classify_que, classify_res_que, config, workerid=0, logger=logging.getLogger("tts")):
        logger.info("start init TtsClassifyWorker:%d init"%(workerid))
        Process.__init__(self)
        self.id = workerid
        self.logger = logger
        model_config_file = "/search/odin/liruihong/tts/multi_attn_model/config_data/classify_config.json"
        model_config_file = config["model_config_file"]
        run_config = {
            "max_seq_length":128,
            "batch_size":1,
            "word2vec_file":"/search/odin/liruihong/tts/multi_attn_model/config_data/100000-small.txt",
            "stop_words_file":"/search/odin/liruihong/tts/multi_attn_model/config_data/cn_stopwords.txt"
        }
        run_config = {
            "max_seq_length":config["max_seq_length"],
            "batch_size":config["batch_size"],
            "word2vec_file":config["word2vec_file"],
            "stop_words_file":config["stop_words_file"]
        }
        processor =  TtsProcessor()
        self.model_server = ModelServer(model_config_file, run_config, processor)

        init_checkpoint = "/search/odin/liruihong/tts/bert_output/wordvec_attn/annotate_part_unlimitlen/model.ckpt-4600"
        model_output_dir = "/search/odin/liruihong/tts/bert_output/wordvec_attn/annotate_part_unlimitlen"
        init_checkpoint = config["init_checkpoint"]
        model_output_dir = config["model_output_dir"]
        self.logger.info("Tts worker[%d] start build model"%(self.id))
        self.model_server.build_model(init_checkpoint, model_output_dir)
        self.ready_to_classify_que = ready_to_classify_que
        self.classify_res_que = classify_res_que
        self.logger.info("finish TtsClassifyWorker:%d init"%(self.id))

    def run(self):
        self.logger.info("TtsClassifyWorker %d start run"%(self.id))
        while True:
            data = self.ready_to_classify_que.get()
            self.logger.debug("get data %s"%(data["req_id"]))
            texts = data["texts"]
            tts_labels = self.model_server.predict(texts)
            data["tts_labels"] = tts_labels
            self.classify_res_que.put(data)


def main():
    config = {
        "model_config_file":"/search/odin/liruihong/tts/multi_attn_model/config_data/classify_config.json",
        "max_seq_length":128,
        "batch_size":32,
        "word2vec_file":"/search/odin/liruihong/tts/multi_attn_model/config_data/100000-small.txt",
        "stop_words_file":"/search/odin/liruihong/tts/multi_attn_model/config_data/cn_stopwords.txt",
        "init_checkpoint":"/search/odin/liruihong/tts/bert_output/wordvec_attn/annotate_part_unlimitlen/model.ckpt-4600",
        "model_output_dir":"/search/odin/liruihong/tts/bert_output/wordvec_attn/annotate_part_unlimitlen",
        "http_port":9001,
    }

    logger = set_logger("root", verbose=True, handler=logging.StreamHandler())
    read_to_classify_que = Queue() 
    classify_res_que = Queue()
    http_server = HTTPServer(config, read_to_classify_que, classify_res_que, 1, logger)
    logger.info("start server")
    http_server.start()

    logger.info("finish all start")
    #http_server.join()


if __name__ == "__main__":
    main()
    #test_server = Test()
    #test_server.start()
    #test_server.join()