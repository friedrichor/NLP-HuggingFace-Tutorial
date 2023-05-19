export PRETRAINED_MODEL_NAME_OR_PATH="t5-base"
export TEXT_PREFIX="dialogue: "
export DATA_DIR="partial_dataset"
export SAVE_WEIGHTS_PATH="weights"

python train.py \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH \
    --text_prefix=$TEXT_PREFIX \
    --data_dir=$DATA_DIR \
    --weights_dir=$SAVE_WEIGHTS_PATH \
    --num_train_epochs 10 \
    --train_batch_size 64 \
    --valid_batch_size 8 \
    --learning_rate 5e-5 \
    --weight_decay 0 \
    --num_warmup_steps 1000 \
    --device "cuda:0"

