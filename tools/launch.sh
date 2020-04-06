%cd /content/drive/My\ Drive/colab_tf_od_train_results/models/research/

! PYTHONPATH=${PYTHONPATH}:./:./object_detection:./slim python object_detection/model_main.py \
  --pipeline_config_path="/content/drive/My Drive/colab_tf_od_train_results/models/jockey_detection/model/ssd_mobilenet_v1.config" \
  --model_dir="/content/drive/My Drive/colab_tf_od_train_results/models/jockey_detection/model/train_7" \
  --sample_1_of_n_eval_examples=1 \
  --num_train_steps=200000 \
  --alsologtostderr