CUDA_VISIBLE_DEVICES=0 python VP_code/test.py --name RNN_Swin_4 --model_name RNN_Swin_4 \
                                              --which_iter 200000 --temporal_length 20 --temporal_stride 10 \
                                              --input_video_url ./test_data \
                                              --gt_video_url ./test_data