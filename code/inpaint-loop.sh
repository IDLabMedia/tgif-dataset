#!/bin/bash

cd /project_ghent/tgiif

echo "Install requirements"
bash ./install_requirements.sh

echo "Start script"

inpainting_model_name="sd2"
#inpainting_model_name="sdxl"

#CATEGORIES='person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush'

CATEGORIES=$1
CATEGORIES_NEW=$CATEGORIES # Inpaint same category

USE_PROMPT_FROM_CAPTION=2 # Use both category+caption(changed)
VERBOSE=0
NUM_INFERENCE_STEPS=50
#NUM_INFERENCE_STEPS=1
#NUM_IMAGES_PER_PROMPT=1
NUM_IMAGES_PER_PROMPT=3
SEED=26
MAX_IMAGES=50

python3 inpaint-loop.py \
    --inpainting_model_name $inpainting_model_name --categories "$CATEGORIES" --categories_new "$CATEGORIES_NEW" \
    --use_bbox_mask --use_segmentation_mask \
    --use_prompt_from_caption "$USE_PROMPT_FROM_CAPTION" \
    --verbose "$VERBOSE" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT" \
    --do_nima --do_itm \
    --seed "$SEED" --max_images $MAX_IMAGES

