## Usage
  Make sure that you have downloaded the pretrained SAM and LaMa models. If they are under `./pretrained_models`. Run the demo web via the following command.
  ```
  python app.py \
        --lama_config ./lama/configs/prediction/default.yaml \
        --lama_ckpt ./pretrained_models/big-lama \
        --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth
  ```
