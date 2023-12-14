# python render_emit.py \
#     ../../datasets/neilf/fipt/synthetic/bedroom/train \
#     ../outputs/fipt_bedroom/ \
#     --config_path ./configs/config_fipt_syn.json

# python render_emit.py \
#     ../../datasets/neilf/fipt/synthetic/bathroom/train \
#     ../outputs/fipt_bathroom/ \
#     --config_path ./configs/config_fipt_syn.json

# python render_emit.py \
#     ../../datasets/neilf/fipt/synthetic/kitchen/train \
#     ../outputs/fipt_kitchen/ \
#     --config_path ./configs/config_fipt_syn.json

# python render_emit.py \
#     ../../datasets/neilf/fipt/synthetic/livingroom/train \
#     ../outputs/fipt_livingroom/ \
#     --config_path ./configs/config_fipt_syn.json

python render_emit.py \
    ../../datasets/neilf/fipt/real/classroom \
    ../outputs/fipt_classroom/ \
    --config_path ./configs/config_fipt_real.json

python render_emit.py \
    ../../datasets/neilf/fipt/real/conferenceroom \
    ../outputs/fipt_conferenceroom/ \
    --config_path ./configs/config_fipt_real.json