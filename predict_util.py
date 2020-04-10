#!/usr/bin/env python
# -*-encoding=utf-8-*-
import tensorflow as tf
import csv
import os
import collections
import numpy as np
import modeling
import tokenization
import optimization
import logging
import time


class InputExample(object):
    """A single training/test example for simple sequence classification."""
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self,
                 guid,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.guid = guid
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    def get_train_examples(self, data_path):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_path):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_path):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class TtsProcessor(DataProcessor):
    def get_train_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path), "train")

    def get_dev_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path),
            "dev_matched")

    def get_test_examples(self, data_path):
        """See base class."""
        return self._create_examples(
            self._read_tsv(data_path), "test")

    def get_predict_examples(self, text_list):
        examples =  []
        for idx, data in enumerate(text_list):
            guid = 'pred-%d' % idx
            text_a = tokenization.convert_to_unicode(text_list[idx])
            text_b = None
            label = "1"
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


    def get_labels(self):
        """See base class."""
        return ["1", "2", "3","4"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[1])

            if len(line) > 2:
                text_b = tokenization.convert_to_unicode(line[2])
            else:
                text_b = None

            if set_type == "test":
                label = "1"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        # _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[0: max_seq_length]

    tokens = []
    #segment_ids = []
    for token in tokens_a:
        tokens.append(token)
    #    #segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
    #        #segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens_a)
    segment_ids = [0]*len(input_ids)
    if tokens_b:
        input_ids_b = tokenizer.convert_tokens_to_ids(tokens_b)
        segment_ids_b = [1]*len(input_ids_b)
        input_ids += input_ids_b
        segment_ids += segment_ids_b
    # input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # input_ids = tokens

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]

    feature = InputFeatures(
        guid=example.guid,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def create_model(model_config,
                 is_training,
                 input_ids,
                 input_mask,
                 segment_ids,
                 labels,
                 num_labels,
                 embedding_table=None,
                 use_one_hot_embeddings=False):
    """Creates a classification model."""
    model = modeling.TextClassify(
        config=model_config,
        is_training=is_training,
        input_ids=input_ids,
        embedding_table=embedding_table,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()
    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(model_config,
                     num_labels,
                     init_checkpoint,
                     embedding_table_value,
                     learning_rate=None,
                     num_train_steps=None,
                     num_warmup_steps=None,
                     embedding_table_trainable=False,
                     use_one_hot_embeddings=False):

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        guid = features["guid"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        embedding_table = tf.get_variable("embedding_table",
                                          shape=[model_config.vocab_size, model_config.vocab_vec_size],
                                          trainable=embedding_table_trainable)
        def init_embedding_table(scoffold,sess):
            sess.run(embedding_table.initializer, {embedding_table.initial_value: embedding_table_value})

        scaffold = tf.train.Scaffold(init_fn=init_embedding_table)

        (total_loss, per_example_loss, logits, probabilities) = create_model(model_config,
                                                                             is_training,
                                                                             input_ids,
                                                                             input_mask,
                                                                             segment_ids,
                                                                             label_ids,
                                                                             num_labels,
                                                                             embedding_table,
                                                                             use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}

        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"guid":guid, "probabilities": probabilities, "input_ids":input_ids, "input_mask":input_mask},
            scaffold=scaffold)
        return output_spec

    return model_fn


def load_embedding_table(embedding_table_file):
    # embedding_table = tokenization.load_embedding_table(embedding_table_file)
    vec_table = []
    with open(embedding_table_file) as fp:
        lines = fp.readlines()
        for i,line in enumerate(lines):
            if i == 0:
                continue
            items = line.rstrip().split(' ')
            word = items[0]
            vec = items[1:]
            vec_table.append(vec)
    embedding_table = np.asarray(vec_table)
    return embedding_table


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):

    #def create_int_feature(values):
    #    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    #    return f
    features = []
    for idx, example in enumerate(examples):
        feature = convert_single_example(idx, example, label_list, max_seq_length, tokenizer)
        feat = collections.OrderedDict()
        #feat["input_ids"] = create_int_feature(feature.input_ids)
        #feat["input_mask"] = create_int_feature(feature.input_mask)
        #feat["segment_ids"] = create_int_feature(feature.segment_ids)
        #feat["label_ids"] = create_int_feature([feature.label_id])
        feat["input_ids"] = feature.input_ids
        feat["input_mask"] = feature.input_mask
        feat["segment_ids"] = feature.segment_ids
        feat["label_ids"] = [feature.label_id]
        features.append(feat)
    return features

def input_fn_builder(processor, label_list, max_seq_length, tokenizer, receive_que):

    #data_examples = processor.get_predict_examples(text_list)
    #features = convert_examples_to_features(data_examples, label_list, max_seq_length, tokenizer)
    def generate_fn():
        input_item = receive_que.get()
        #data_examples = processor.get_predict_examples([input_item])
        data_example = InputExample(guid=input_item['guid'],text_a=input_item['text_a'])
        feature = convert_single_example(data_example,label_list,max_seq_length,tokenizer)
        # features = convert_examples_to_features(data_examples, label_list, max_seq_length, tokenizer)
        # for feat in features:
        #    """
        #    feat : {
        #        "input_ids":[],
        #        "input_mask":[],
        #        "segment_ids":[],
        #        "label_ids":[]
        #    }
        #    """
        #    yield feat
        yield feature

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
                "guid":(1),
                "input_ids":(max_seq_length),
                "input_mask":(max_seq_length),
                "segment_ids":(max_seq_length),
                "label_ids":(1)
            }
        )

        feature_data = feature_data.batch(params['batch_size'])
        iter = feature_data.make_one_shot_iterator()
        batch_data = iter.get_next()
        feature_dict = {
            'guid':batch_data['guid'],
            'input_ids':batch_data['input_ids'],
            'input_mask':batch_data['input_mask'],
            'segment_ids':batch_data['segment_ids'],
            'label_ids':batch_data['label_ids']
        }  
        return feature_dict,None

    return input_fn

class ModelServer(object):
    def __init__(self, model_config_file, run_config, processor, logger=logging.getLogger()):
        self.logger = logger
        self.model_config = modeling.ModelConfig.from_json_file(model_config_file)
        self.processor = processor
        self.tokenizer = tokenization.Tokenizer(
            word2vec_file=run_config["word2vec_file"], stop_words_file=run_config["stop_words_file"])
        self.label_list = self.processor.get_labels()
        self.run_config = run_config
        self.params = {
            "max_seq_length":run_config["max_seq_length"],
            "batch_size":run_config["batch_size"]
        }
        self.embedding_table = load_embedding_table(run_config["word2vec_file"])

    def build_model(self, init_checkpoint, model_output_dir):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess_config.log_device_placement = False
        run_config = tf.estimator.RunConfig(session_config=sess_config)
        model_fn = model_fn_builder(
            model_config=self.model_config,
            num_labels=len(self.label_list),
            init_checkpoint=init_checkpoint,
            embedding_table_value=self.embedding_table)

        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            model_dir=model_output_dir,
            params=self.params) 
        tf.logging.info("finish model building")

    def predict(self, receive_que, output_predict_file=None):
        input_fn = input_fn_builder(self.processor, self.label_list, self.run_config["max_seq_length"], self.tokenizer, receive_que)
        self.logger.debug("prepare input_fn, start predict")
        result = self.estimator.predict(input_fn=input_fn, yield_single_examples=True) 
        if output_predict_file is None:
            return result
            #for (i, prediction) in enumerate(result):
            #    probabilities = prediction["probabilities"]
            #    pred_label = np.argmax(probabilities) + 1
            #    #pred_res.append(pred_label)
            #    pred_res_que.put()
        else:
            with tf.gfile.GFile(output_predict_file, "w") as writer:
                num_written_lines = 0
                tf.logging.info("***** Predict results *****")
                for (i, prediction) in enumerate(result):
                    probabilities = prediction["probabilities"]
                    input_ids = prediction["input_ids"]
                    output_line = "\t".join(
                        str(class_probability)
                        for class_probability in probabilities) + "\t"
                    words_str = " ".join(self.tokenizer.convert_ids_to_tokens(input_ids)) + "\n"
                    writer.write(output_line + words_str)
                    num_written_lines += 1

def load_predict_file(file_name):
    text_list = []
    with open(file_name,"r") as fp:
        lines = fp.readlines()
        for i,line in enumerate(lines):
            if i == 0:
                continue
            text = line.strip().split('\t')[1]
            text_list.append(text)
    return text_list



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format="[%(asctime)s-%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    model_config_file = "/search/odin/liruihong/tts/multi_attn_model/config_data/classify_config.json"
    run_config = {
        "max_seq_length":128,
        "batch_size":1,
        "word2vec_file":"/search/odin/liruihong/tts/multi_attn_model/config_data/100000-small.txt",
        "stop_words_file":"/search/odin/liruihong/tts/multi_attn_model/config_data/cn_stopwords.txt"
    }
    processor =  TtsProcessor()
    model_server = ModelServer(model_config_file, run_config, processor)
    
    init_checkpoint = "/search/odin/liruihong/tts/bert_output/wordvec_attn/annotate_part_unlimitlen/model.ckpt-4600"
    model_output_dir = "/search/odin/liruihong/tts/bert_output/wordvec_attn/annotate_part_unlimitlen"
    model_server.build_model(init_checkpoint, model_output_dir)

    data_file = "/search/odin/liruihong/tts/data/eval_data/sample_31d"
    text_list = load_predict_file(data_file) 

    st = time.time()*1000
    model_server.predict(text_list)
    ed = time.time()*1000
    cost = int(ed - st)
    logging.info("[time cost] %d ms"%(cost))   