# Data storage structure

## DATA root directory
Assign the variable WUW_DATA_ROOT to the root directory of the data.

`DATA_ROOT=os.environ["WUW_DATA_ROOT"]`

## METADATA

The metadata format adheres to Kaldi's format.

The format of the index file (labels.index) is as follows.

`<uttid> <label> <end_timestamp> <relative_path_to_wav_fn>`

`absolute_path = DATA_ROOT + "/" + relative_path_to_wav_fn`

## AudioLoader

AudioLoader configuration

**model.yaml**
<pre>
data_conf:
  data_dir: /data/smartspeaker/release/
  train_index: train/trainsetX/train_labels.index
  val_index: train/trainsetX/val_labels.index
  test_index: test/testsetX/labels.index
feat_conf:
  window_size_ms: 40.0
  window_stride_ms: 20.0
  dct_coefficient_count: 10
  upper_frequency_limit: 8000
  lower_frequency_limit: 20
  filterbank_channel_count: 40
model_conf:
  name: xxxx
  training_steps: [ 2000, 5000, 3000 ]
  eval_step_interval: 100
  learning_rate: [ 0.00005, 0.00003, 0.00001 ]
  batch_size: 200
  
  model_size_info: [ ... ]
</pre>
  





