#!/bin/sh

BERT_BASE_DIR=/search/odin/zhuguangnan/bert_test/bert_data/uncased_L-12_H-768_A-12
BERT_ZH_DIR=/search/odin/zhuguangnan/bert_test/bert_data/chinese_L-12_H-768_A-12
TMP_DIR=/search/odin/zhuguangnan/bert_test/tmpdata

DATA_DIR=/search/odin/liruihong/tts/data
word2vec="/search/odin/liruihong/tts/multi_attn_model/config_data/70000-small.txt"
config_file="/search/odin/liruihong/tts/multi_attn_model/config_data/classify_config.json"

test_data=$DATA_DIR/eval_data/test_4label_review_v2.tsv 
#test_data=$DATA_DIR/eval_data/test100_bertinput
train_data=$DATA_DIR/train_data/annotate_augment2_train
dev_data=$DATA_DIR/train_data/annotate_data_dev
#init_dir=$TMP_DIR/zh_topic_sub1kw_pretrain_output/model.ckpt-50 
#--init_checkpoint=$init_dir \

python run_textclassify.py \
  --task_name=tts \
  --do_train=true \
  --do_eval=false \
  --do_predict=false \
  --data_path=$test_data \
  --word2vec_file=$word2vec \
  --config_file=$config_file \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=6 \
  --output_dir=$OUTPUT_DIR
