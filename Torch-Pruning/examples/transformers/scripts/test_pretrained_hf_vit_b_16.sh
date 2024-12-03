python finetune.py \
    --model "google/vit-base-patch16-224" \
    --epochs 300 \
    --batch-size 32 \
    --opt adamw \
    --lr 0.00015 \
    --wd 0.3 \
    --lr-scheduler cosineannealinglr \
    --lr-warmup-method linear \
    --lr-warmup-epochs 0 \
    --lr-warmup-decay 0.033 \
    --amp \
    --label-smoothing 0.11 \
    --mixup-alpha 0.2 \
    --auto-augment ra \
    --clip-grad-norm 1 \
    --ra-sampler \
    --cutmix-alpha 1.0 \
    --data-path "data/imagenet" \
    --test-only \
    --interpolation bilinear \
    --is_huggingface \