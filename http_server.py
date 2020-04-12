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
import zmq
import zmq.decorators as zmqd
from zmq.utils import jsonapi
import numpy as np
import os
import tensorflow as tf
import modeling
import tokenization


class Test(Process):
    def __init__(self):
        super(Process, self).__init__()

    def run(self):
        print("work")
        time.sleep(3)

class DataItem(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class HTTPServer(object):
    def __init__(self, config, ready_to_classify_que, classify_res_que, num_worker=1, send_port=5664, recv_port=5665, logger=logging.getLogger("sver")):
        #super(Process,self).__init__()
        #Process.__init__(self)
        self.config = config
        self.is_ready = Event()
        self.classify_res = dict()
        self.ready_to_classify_que = ready_to_classify_que
        self.classify_res_que = classify_res_que
        self.num_worker = num_worker
        self.logger=logger
        self.send_port=send_port
        self.recv_port=recv_port
        send_context = zmq.Context()
        self.sender = send_context.socket(zmq.PUSH) 
        self.sender.bind("tcp://127.0.0.1:%d"%(send_port))

        recv_context = zmq.Context()
        self.receiver = recv_context.socket(zmq.PULL)
        self.receiver.bind("tcp://127.0.0.1:%d"%(recv_port))
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
            for k in range(req_num):
                textid = uuid.uuid1().hex
                text_a = texts[k]
                text_b = None
                # input_item = DataItem(textid,text_a,None,None)
                text_ids.append(textid)

                input_item = {
                    "guid":textid,
                    "text_a":text_a,
                    "text_b":text_b,
                }
                self.sender.send_json(jsonapi.dumps(input_item))
                # self.ready_to_classify_que.put(input_item)
                self.logger.debug("put item:%s"%(textid))

            req_time = time.time()
            req_data['req_time'] = req_time
            # self.ready_to_classify_que.put(req_data)

            collect_num = 0
            pred_labels = [0]*req_num
            for k in range(req_num):
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
            #if self.classify_res_que.empty():            
            #    continue
            #else:
            #    try:
            #        data_item = self.classify_res_que.get_nowait()
            #    except:
            #        continue
            #    self.logger.debug('put %s res back'%(data_item['req_id']))
            #    self.classify_res[data_item['req_id']] = data_item
            events = self.receiver.poll()
            if events:
                data_item = self.receiver.recv_json()
                data_item = jsonapi.loads(data_item)
                guid = data_item['guid']
                self.logger.debug('put %s res back'%(guid))
                self.classify_res[guid] = data_item['label']

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
            tts_server = TtsClassifyWorker(self.ready_to_classify_que, self.classify_res_que, self.config, send_port=self.recv_port, recv_port=self.send_port, workerid=i, logger=self.logger)
            tts_server.start()
        

class TtsClassifyWorker(Process):
    def __init__(self, ready_to_classify_que, classify_res_que, config, send_port, recv_port, workerid=0, logger=logging.getLogger("tts")):

        logger.info("start init TtsClassifyWorker:%d init"%(workerid))
        Process.__init__(self)
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(workerid)
        self.id = workerid
        self.logger = logger
        self.send_port=send_port
        self.recv_port=recv_port
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
        self.processor =  TtsProcessor()
        self.model_config = modeling.ModelConfig.from_json_file(model_config_file)
        self.tokenizer = tokenization.Tokenizer(
            word2vec_file=run_config["word2vec_file"], stop_words_file=run_config["stop_words_file"])
        self.label_list = self.processor.get_labels()
        self.run_config = run_config
        self.params = {
            "max_seq_length":run_config["max_seq_length"],
            "batch_size":run_config["batch_size"]
        }
        self.embedding_table = load_embedding_table(run_config["word2vec_file"])
        # self.model_server = ModelServer(model_config_file, run_config, processor, self.logger)

        init_checkpoint = "/search/odin/liruihong/tts/bert_output/wordvec_attn/annotate_part_unlimitlen/model.ckpt-4600"
        model_output_dir = "/search/odin/liruihong/tts/bert_output/wordvec_attn/annotate_part_unlimitlen"
        self.init_checkpoint = config["init_checkpoint"]
        self.model_output_dir = config["model_output_dir"]
        self.logger.info("Tts worker[%d] start build model"%(self.id))
        #self.model_server.build_model(init_checkpoint, model_output_dir)
        #self.get_estimator()
        self.ready_to_classify_que = ready_to_classify_que
        self.classify_res_que = classify_res_que
        self.logger.info("finish TtsClassifyWorker:%d init"%(self.id))
    
    def get_estimator(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess_config.log_device_placement = False
        run_config = tf.estimator.RunConfig(session_config=sess_config)
        model_fn = model_fn_builder(
            model_config=self.model_config,
            num_labels=len(self.label_list),
            init_checkpoint=self.init_checkpoint,
            embedding_table_value=self.embedding_table)

        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            model_dir=self.model_output_dir,
            params=self.params) 
        self.logger.info("finish model building")
        return self.estimator

    def collect_pred_res(self, pred_generator):
        for pred_res in pred_generator:
            pred_label = np.argmax(pred_res["probabilities"]) + 1
            guid = pred_res["guid"]
            self.logger.debug("collect %s, pred:%d"%(guid, pred_label)) 
            data_item = DataItem(guid=guid,text_a=None,text_b=None,label=pred_label)
            self.classify_res_que.put(data_item)

    def run(self):
        self.logger.info("TtsClassifyWorker %d start run"%(self.id))
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PUSH)
    def _run(self, _, receiver, sender):
        estimator = self.get_estimator()
        self.logger.info('bind all sockets')
        receiver.connect('tcp://127.0.0.1:%d'%(self.recv_port))
        sender.connect('tcp://127.0.0.1:%d'%(self.send_port))
        # pred_res_generator = self.model_server.predict()
        res_generator = estimator.predict(input_fn=self.input_fn_builder(receiver, self.label_list, self.run_config['max_seq_length'], self.tokenizer), yield_single_examples=True)
        for res_item in res_generator:
            guid = res_item['guid']
            self.logger.debug("worker[%d] get pred res %s"%(self.id, guid))
            pred_label = np.argmax(res_item['probabilities']) + 1
            data_item = {
               "guid":guid,
               "label":pred_label 
            }
            sender.send_json(jsonapi.dumps(data_item))
        # collect_res_thread = threading.Thread(target=self.collect_pred_res, args=(pred_res_generator))
        # collect_res_thread.start()
        # collect_res_thread.join()

        #while True:
        #    data_item = self.ready_to_classify_que.get()
        #    self.logger.debug("get data %s"%(data_item["guid"]))
        #    # tts_labels = self.model_server.predict(texts)
        #    # data["tts_labels"] = tts_labels
        #    self.logger.debug("get predict res req_id:%s, res:%s, cost:%d ms"%(data["req_id"], str(tts_labels), cost))
        #    # self.classify_res_que.put(data)

    def input_fn_builder(self, receiver_sock, label_list, max_seq_length, tokenizer):

        def generate_fn():
            #poller = zmq.Poller()
            #poller.register(receiver_sock, zmq.POLLIN)
            while True:
                events = receiver_sock.poll()
                if events:
                    data_item = receiver_sock.recv_json()
                    data_item = jsonapi.loads(data_item)
                    data_example = InputExample(guid=data_item['guid'],text_a=data_item['text_a'],text_b=data_item['text_b'],label="1")
                    feature = convert_single_example(data_example,label_list,max_seq_length,tokenizer)
                    self.logger.debug("input_fn yield %s"%(feature.guid))
                    yield {
                        "guid":[feature.guid],
                        "input_ids":[feature.input_ids],
                        "input_mask":[feature.input_mask],
                        "segment_ids":[feature.segment_ids],
                        "label_ids":[[feature.label_id]]
                    }
        
        def input_fn(params):
            max_seq_length = params["max_seq_length"]
            feature_data = tf.data.Dataset.from_generator(
                generate_fn,
                output_types={
                    "guid":tf.string,
                    "input_ids":tf.int32,
                    "input_mask":tf.int32,
                    "segment_ids":tf.int32,
                    "label_ids":tf.int32
                },
                output_shapes={
                    "guid":(None),
                    "input_ids":(None,max_seq_length),
                    "input_mask":(None,max_seq_length),
                    "segment_ids":(None,max_seq_length),
                    "label_ids":(None,1)
                }
            )

            #feature_data = feature_data.batch(params['batch_size'])
            #iter = feature_data.make_one_shot_iterator()
            #batch_data = iter.get_next()
            #feature_dict = {
            #    'guid':batch_data['guid'],
            #    'input_ids':batch_data['input_ids'],
            #    'input_mask':batch_data['input_mask'],
            #    'segment_ids':batch_data['segment_ids'],
            #    'label_ids':batch_data['label_ids']
            #}  
            #return feature_dict,None
            return feature_data
        return input_fn
    
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
    http_server = HTTPServer(config, ready_to_classify_que, classify_res_que, 2, 5664,5665, logger)
    logger.info("start server")
    http_server.start()


if __name__ == "__main__":
    main()
    #test_server = Test()
    #test_server.start()
    #test_server.join()