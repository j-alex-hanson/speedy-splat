data_path=$SCENE_DATA_PATH
model_path=$SCENE_MODEL_PATH
kernel_times=$ONLY_RAW_KERNEL_TIMES

if [ "$ONLY_RAW_KERNEL_TIMES" = "true" ]; then

  python compute_scene_metrics.py \
    -s ${data_path} \
    -m ${model_path} \
    --eval \
    --kernel_times

else

  python compute_scene_metrics.py \
    -s ${data_path} \
    -m ${model_path} \
    --eval

fi


