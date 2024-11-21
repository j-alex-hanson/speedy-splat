data_path=$SCENE_DATA_PATH
model_path=$SCENE_MODEL_PATH

python train.py \
  -s ${data_path} \
  -m ${model_path} \
  --eval

