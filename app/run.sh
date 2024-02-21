export TRANSFORMERS_OFFLINE=1;
export HF_DATASETS_OFFLINE=1;
export HF_HOME=~/scratch/cache/;
export HF_DATASETS_CACHE=~/scratch/cache/;
export TRANSFORMERS_CACHE=~/scratch/cache/;
python app_refactor.py \
      --lama_config /home/hice1/mnigam9/scratch/meher/editing/EquiFash/lama/configs/prediction/default.yaml \
      --lama_ckpt ./pretrained_models/big-lama \
      --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth