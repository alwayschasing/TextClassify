#!/usr/bin/env python
# -*- encoding:utf-8 -*-

import sys
from multiprocessing import Process, Event, Queue
import time
import threading
import logging
import uuid
from predict_util import *
from helper import set_logger

class Test(Process):
    def __init__(self):
        super(Process, self).__init__()

    def run(self):
        print("work")
        time.sleep(3)

class DataItem(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = uuid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class HTTPServer(object):
    def __init__(self, config, ready_to_classify_que, classify_res_que, num_worker=1, logger=logging.getLogger("sver")):
        #super(Process,self).__init__()
        Process.__init__(self)
        self.config = config
        self.is_ready = Event()
        self.classify_res = dict()
        self.ready_to_classify_que = ready_to_classify_que
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
        self.create_classify_worker()

        @app.route('/tts-classify', methods=['POST'])
        @as_json
        def tts_classify():
            req_data = request.form if request.form else request.json
            # req_data['req_id'] = uuid.uuid1()
            # req_id = req_data['req_id']
            texts = req_data['texts'] # text list
            if not isinstance(texts, list):
                texts = [texts]
            req_num = len(texts)
            text_ids = []
            for k in req_num:
                textid = uuid.uuid1()
                text_a = texts[k]
                input_item = DataItem(textid,text_a,None,None)
                text_ids.append(textid)

            req_time = time.time()
            req_data['req_time'] = req_time
            self.logger.debug("request %s"%(texts[0]))
            self.ready_to_classify_que.put(req_data)

            collect_num = 0
            pred_labels = [0]*req_num
            for k in req_num:
                while text_ids[k] not in self.classify_res:
                    continue
                pred_labels[k] = self.classify_res[text_ids[k]]
                self.classify_res.pop(text_ids[k])

            res_data = {
                "pred_labels":pred_labels
            }
            return res_data
        # CORS(app, origins=self.args.cors)
        FlaskJSON(app)
        Compress().init_app(app)
        return app

    def collect_worker_res(self):
        while True:
            if self.classify_res_que.empty():            
                continue
            else:
                try:
                    data_item = self.classify_res_que.get_nowait()
                except:
                    continue
                self.logger.debug('put %s res back'%(data_item['req_id']))
                self.classify_res[data_item['req_id']] = data_item

    def start(self):
        # 启动分类结果接收线程
        self.logger.info("start run")
        receive_thread = threading.Thread(target=self.collect_worker_res)
        receive_thread.start()

        self.logger.info("start create app")
        app = self.create_flask_app()
        self.is_ready.set()
        app.run(port=self.config["http_port"], threaded=True, host='0.0.0.0')
        self.logger.info("list to port:%d"%(self.config["http_port"]))
        receive_thread.join()

    def create_classify_worker(self):
        for i in range(self.num_worker):
            tts_server = TtsClassifyWorker(self.ready_to_classify_que, self.classify_res_que, self.config, workerid=i, logger=self.logger)
            tts_server.start()
        

class TtsClassifyWorker(Process):
    def __init__(self, ready_to_classify_que, classify_res_que, config, workerid=0, logger=logging.getLogger("tts")):
        logger.info("start init TtsClassifyWorker:%d init"%(workerid))
        Process.__init__(self)
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(workerid)
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
        self.model_server = ModelServer(model_config_file, run_config, processor, self.logger)

        init_checkpoint = "/search/odin/liruihong/tts/bert_output/wordvec_attn/annotate_part_unlimitlen/model.ckpt-4600"
        model_output_dir = "/search/odin/liruihong/tts/bert_output/wordvec_attn/annotate_part_unlimitlen"
        init_checkpoint = config["init_checkpoint"]
        model_output_dir = config["model_output_dir"]
        self.logger.info("Tts worker[%d] start build model"%(self.id))
        self.model_server.build_model(init_checkpoint, model_output_dir)
        self.ready_to_classify_que = ready_to_classify_que
        self.classify_res_que = classify_res_que
        self.logger.info("finish TtsClassifyWorker:%d init"%(self.id))

    def collect_pred_res(self, pred_generator):
        for pred_res in pred_generator:
            pred_label = np.argmax(pred_res["probabilities"]) + 1
            guid = pred_res["guid"]
            self.logger.debug("collect %s, pred:%d"%(guid, pred_label)) 
            data_item = DataItem(guid=guid,text_a=None,text_b=None,label=pred_label)
            self.classify_res_que.put(data_item)

    def run(self):
        self.logger.info("TtsClassifyWorker %d start run"%(self.id))
        pred_res_generator = self.model_server.predict(self.ready_to_classify_que)
        collect_res_thread = threading.Thread(target=self.collect_pred_res, args=(pred_res_generator))
        collect_res_thread.start()
        collect_res_thread.join()

        #while True:
        #    data_item = self.ready_to_classify_que.get()
        #    self.logger.debug("get data %s"%(data_item["guid"]))
        #    # tts_labels = self.model_server.predict(texts)
        #    # data["tts_labels"] = tts_labels
        #    self.logger.debug("get predict res req_id:%s, res:%s, cost:%d ms"%(data["req_id"], str(tts_labels), cost))
        #    # self.classify_res_que.put(data)


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
    ready_to_classify_que = Queue() 
    classify_res_que = Queue()
    http_server = HTTPServer(config, ready_to_classify_que, classify_res_que, 1, logger)
    logger.info("start server")
    http_server.start()


if __name__ == "__main__":
    main()
    #test_server = Test()
    #test_server.start()
    #test_server.join()