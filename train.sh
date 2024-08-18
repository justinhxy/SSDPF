python train_05.py --data_dir=/mntcephfs/data/med/penet \
                --save_dir=/mntcephfs/lab_data/wangcm/wangzhipeng/05 \
                --ckpt_path=/mntcephfs/lab_data/wangcm/wangzhipeng/05/test_1/best.pth.tar \
                \
                --name=test \
                --model=SSPre1_3 \
                --batch_size=4 \
                --gpu_ids=0 \
                --iters_per_print=8 \
                --iters_per_visual=8000 \
                --learning_rate=1e-2 \
                --lr_decay_step=600000 \
                --lr_scheduler=cosine_warmup \
                --num_epochs=20 \
                --num_slices=32 \
                --weight_decay=1e-3 \
                \
                --phase=train \
                \
                --abnormal_prob=0.3 \
                --agg_method=max \
                --best_ckpt_metric=val_loss \
                --crop_shape=192,192 \
                --cudnn_benchmark=False \
                --dataset=pe \
                --do_classify=True \
                --epochs_per_eval=1 \
                --epochs_per_save=1 \
                --fine_tune=False \
                --fine_tuning_boundary=classifier \
                --fine_tuning_lr=1e-2 \
                --include_normals=True \
                --lr_warmup_steps=10000 \
                --model_depth=50 \
                --num_classes=1 \
                --num_visuals=8 \
                --num_workers=4 \
                --optimizer=sgd \
                --pe_types='["central","segmental"]' \
                --resize_shape=208,208 \
                --sgd_dampening=0.9 \
                --sgd_momentum=0.9 \
                --use_hem=False \
                --use_pretrained=False \
                
                
		
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
