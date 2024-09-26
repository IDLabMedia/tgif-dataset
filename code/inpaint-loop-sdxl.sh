#!/bin/bash

cd /project_ghent/tgiif

echo "Install requirements"
bash ./install_requirements.sh

echo "Start script"
inpainting_model_name="sdxl"

#CATEGORIES='person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush'
CATEGORIES='person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush' # without frisbee,knife
#CATEGORIES='cat'
#CATEGORIES=$1

CATEGORIES_NEW=$CATEGORIES # Inpaint same category

USE_PROMPT_FROM_CAPTION=2 # Use both category+caption(changed)
VERBOSE=0
MAX_IMAGES=50

NUM_IMAGES_PER_PROMPT=3

SAVE_DIR=output_${inpainting_model_name}
mkdir -p $SAVE_DIR

SAVE_ORIG_DIR=output_${inpainting_model_name}_orig
mkdir -p $SAVE_ORIG_DIR
#SAVE_ORIG_DIR=

BAK_ORIG_DIR=output_${inpainting_model_name}_orig
#BAK_ORIG_DIR=

PS_MASK_DIR=ps_masks
ps_cmd="--use_ps_mask_for_splicing --ps_mask_dir "$PS_MASK_DIR""
#ps_cmd=""

skip_cmd=
#SKIP_DIR=output_${inpainting_model_name}
#skip_cmd="--skip_if_exists --skip_dir "$SKIP_DIR""

stop_on_error_cmd=
#stop_on_error_cmd=--stop_on_exception

#nima_itm_cmd=
nima_itm_cmd="--do_nima --do_itm --do_giqa"

python3 inpaint-loop.py \
    --inpainting_model_name $inpainting_model_name --categories "$CATEGORIES" --categories_new "$CATEGORIES_NEW" \
    --save_dir "$SAVE_DIR" --save_orig_dir "$SAVE_ORIG_DIR" --bak_orig_dir "$BAK_ORIG_DIR" \
    --use_bbox_mask --use_segm_mask \
    --use_prompt_from_caption "$USE_PROMPT_FROM_CAPTION" $ps_cmd \
    --verbose "$VERBOSE" \
    --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT" $nima_itm_cmd $skip_cmd $fix_seed_cmd \
    --max_images $MAX_IMAGES $stop_on_error_cmd

echo "Inpaint loop done"
