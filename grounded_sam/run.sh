# export TRANSFORMERS_OFFLINE=1;
# export HF_DATASETS_OFFLINE=1;
export HF_HOME=~/scratch/cache/;
export HF_DATASETS_CACHE=~/scratch/cache/;
export TRANSFORMERS_CACHE=~/scratch/cache/;
python gsam_demo.py \
  --config ./dino_config_swin_OGC.py \
  --grounded_checkpoint ./weights/groundingdino_swint_ogc.pth \
  --sam_checkpoint /home/hice1/mnigam9/scratch/meher/editing/EquiFash/pretrained_models/sam_vit_h_4b8939.pth \
  --input_image model.png \
  --output_dir "outputs" \
  --box_threshold 0.3 \
  --text_threshold 0.25 \
  --det_prompt "blue dress" \
  --inpaint_prompt "black dress with sequins" \
  --device "cuda"