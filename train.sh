# DIFFSYNTH_ATTENTION_IMPLEMENTATION=flash_attention_2 \
DIFFSYNTH_SKIP_DOWNLOAD=true \ 
CUDA_VISIBLE_DEVICES=0,1 \
accelerate launch examples/flux2/model_training/train.py \
  --dataset_base_path /data3/jiangyikun/AIGC/DiffSynth-Studio/datasets/OmniStyle-150k/ \
  --dataset_metadata_path /data3/jiangyikun/AIGC/DiffSynth-Studio/datasets/style_transfer/OmniStyle-150k/metadata.json \
  --data_file_keys "image,edit_image" \
  --dataset_num_workers 16 \
  --extra_inputs "edit_image" \
  --max_pixels 1048576 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B:text_encoder/*.safetensors,/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B:transformer/*.safetensors,/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B:vae/diffusion_pytorch_model.safetensors" \
  --tokenizer_path "/data3/jiangyikun/AIGC/models/FLUX.2-klein-4B:tokenizer/" \
  --learning_rate 1e-4 \
  --num_epochs 2 \
  --save_steps 2000 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/style_transfer/FLUX.2-klein-4B_lora" \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out,linear_in,linear_out,to_qkv_mlp_proj,single_transformer_blocks.0.attn.to_out,single_transformer_blocks.1.attn.to_out,single_transformer_blocks.2.attn.to_out,single_transformer_blocks.3.attn.to_out,single_transformer_blocks.4.attn.to_out,single_transformer_blocks.5.attn.to_out,single_transformer_blocks.6.attn.to_out,single_transformer_blocks.7.attn.to_out,single_transformer_blocks.8.attn.to_out,single_transformer_blocks.9.attn.to_out,single_transformer_blocks.10.attn.to_out,single_transformer_blocks.11.attn.to_out,single_transformer_blocks.12.attn.to_out,single_transformer_blocks.13.attn.to_out,single_transformer_blocks.14.attn.to_out,single_transformer_blocks.15.attn.to_out,single_transformer_blocks.16.attn.to_out,single_transformer_blocks.17.attn.to_out,single_transformer_blocks.18.attn.to_out,single_transformer_blocks.19.attn.to_out" \
  --lora_rank 32 \
  --enable_tensorboard_log \
#   --use_gradient_checkpointing 			


