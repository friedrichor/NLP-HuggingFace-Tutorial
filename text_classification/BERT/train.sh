export PRETRAINED_MODEL_NAME_OR_PATH="bert-base-uncased"
export DATA_DIR="dataset"
export WEIGHTS_DIR="weights"

python train.py \
    --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH \
    --data_dir=$DATA_DIR \
    --weights_dir=$WEIGHTS_DIR \
    --num_train_epochs 10 \
    --train_batch_size 64 \
    --valid_batch_size 32 \
    --learning_rate 5e-4 \
    --weight_decay 0 \
    --num_warmup_steps 500 \
    --device "cuda:0"
