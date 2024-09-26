from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
import torch
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
import skimage.io as io
import os
import datetime
import traceback
import csv
from typing import Callable, Dict
from urllib.error import HTTPError
import argparse

from NIMA import nima
from GIQA import gmm_score
from transformers import BlipProcessor, BlipForImageTextRetrieval

def show_image(img, figsize=(8,8)):
    fig, ax = plt.subplots()
    ax.imshow(np.array(img).astype(np.uint8))
    plt.show()

def resize_image(image, resize_to):
    return np.array(Image.fromarray(image).resize(resize_to, resample=Image.NEAREST))

def ensure_dir_from_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
def ensure_dir_from_file(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
def create_mask_from_bbox(bbox, image_size):
    """
    Create a Luminance (1-channel) image with a white rectangle at the specified bounding box location.

    Parameters:
    - bbox (list): Bounding box coordinates [x, y, width, height].
    - image_size (tuple): Image dimensions (width, height).

    Returns:
    - image_np (np.array): The created image as numpy array.
    """
    background_color = 0 # black
    image = Image.new('L', image_size, color=background_color)
    draw = ImageDraw.Draw(image)

    x, y, width, height = map(int, bbox)
    box_color = 255 # white
    draw.rectangle([x, y, x + width, y + height], fill=box_color)

    image_np = np.array(image)

    return image_np

def merge_bboxes(bboxes):
    """
    Merge multiple bounding boxes into a single bounding box that covers all of them.

    Parameters:
    - bboxes (list): List of bounding boxes, where each bounding box is represented as [x, y, width, height].

    Returns:
    - list: Merged bounding box coordinates [x, y, width, height].
    """
    # Initialize min and max coordinates with the first bounding box
    first_bbox = bboxes[0]
    min_x, min_y, max_x, max_y = first_bbox[0], first_bbox[1], first_bbox[0] + first_bbox[2], first_bbox[1] + first_bbox[3]

    # Iterate through the rest of the bounding boxes to find the min and max coordinates
    for bbox in bboxes[1:]:
        min_x = min(min_x, bbox[0])
        min_y = min(min_y, bbox[1])
        max_x = max(max_x, bbox[0] + bbox[2])
        max_y = max(max_y, bbox[1] + bbox[3])

    # Calculate the width and height of the merged bounding box
    width = max_x - min_x
    height = max_y - min_y

    return [min_x, min_y, width, height]

def create_and_crop_larger_bbox_around_bbox(img, bbox, new_bbox_width, new_bbox_height, divisible_by=8):
    """
    Create and crop a larger bounding box around a given bounding box, ensuring the old bounding box is centered as much as possible.

    Parameters:
    - img (numpy.ndarray): Input image as a numpy array.
    - bbox (list): Bounding box coordinates [x, y, width, height].
    - new_bbox_width (int): Width of the new bounding box.
    - new_bbox_height (int): Height of the new bounding box.

    Returns:
    - numpy.ndarray: Cropped image based on the new bounding box.
    """
    # Extract old bounding box coordinates
    x, y, width, height = bbox

    # Calculate the top-left corner of the new bounding box
    new_x = max(0, x + width / 2 - new_bbox_width / 2)
    new_y = max(0, y + height / 2 - new_bbox_height / 2)

    # Ensure the new bounding box stays within the image boundaries
    image_width, image_height = img.shape[1], img.shape[0]
    if new_bbox_width < image_width:
        new_x = min(new_x, image_width - new_bbox_width)
    else:
        new_x = 0
        new_bbox_width = (image_width // divisible_by) * divisible_by
    if new_bbox_height < image_height:
        new_y = min(new_y, image_height - new_bbox_height)
    else:
        new_y = 0
        new_bbox_height = (image_height // divisible_by) * divisible_by
        
    # Perform the crop based on the new bounding box coordinates
    cropped_image = img[int(new_y):int(new_y) + new_bbox_height, int(new_x):int(new_x) + new_bbox_width]

    # Define the new bounding box coordinates
    new_bbox = [int(new_x), int(new_y), new_bbox_width, new_bbox_height]

    return cropped_image, new_bbox

def get_bbox_with_padding(image_width, image_height, bbox, padding, divisible_by=8):
    """
    Calculate the coordinates of a new bounding box with padding on all sides, considering the image boundaries.

    Parameters:
    - image_width (int): Width of the image.
    - image_height (int): Height of the image.
    - bbox (list): Bounding box coordinates [x, y, width, height].
    - padding (int): Padding value.
    - divisible_by (int): output width and height will be multiple of this

    Returns:
    - tuple: New bounding box coordinates (x, y, width, height) with padding.
    """
    x, y, width, height = bbox

    # Calculate new bounding box coordinates with padding
    x_new = max(0, x - padding)
    y_new = max(0, y - padding)
    x_new, y_new = int(x_new), int(y_new)
    width_new = min(image_width - x_new, width + 2 * padding)
    height_new = min(image_height - y_new, height + 2 * padding)
    width_new, height_new = int(width_new), int(height_new)
    print(x_new, y_new, width_new, height_new)
    x_new, width_new = make_divisible_by(x_new, width_new, divisible_by, image_width)
    y_new, height_new = make_divisible_by(y_new, height_new, divisible_by, image_height)
             
    return x_new, y_new, width_new, height_new

def make_divisible_by(x, width, divisible_by, image_width):
    if width % divisible_by != 0:
        extra_needed = divisible_by - (width % divisible_by) # 6

        if width + extra_needed >= image_width: # Make smaller rather than larger
            #print("Deleting pixels!")
            pixels_to_delete = (width % divisible_by)
            pixels_to_delete_left = math.floor(pixels_to_delete / 2)
            pixels_to_delete_right = math.ceil(pixels_to_delete / 2)
            new_x = x + pixels_to_delete_left
            new_width = width - pixels_to_delete_right - pixels_to_delete_left
        else: # Make larger
            new_x = x
            new_width = width
            extra_needed_left = math.floor(extra_needed / 2)
            extra_needed_right = math.ceil(extra_needed / 2)

            if x + width + extra_needed_right > image_width: # Not enough space on the right
                extra_to_transfer = (x + width + extra_needed_right) - image_width
                extra_needed_left += extra_to_transfer
                extra_needed_right -= extra_to_transfer
    
            if x - extra_needed_left < 0: # Not enough space on the left
                extra_to_transfer = extra_needed_left - x
                extra_needed_right += extra_to_transfer
                extra_needed_left -= extra_to_transfer

            if extra_needed_right > 0: # Add the not-enough-space-on-the-left to the right
                new_width += extra_needed_right
            if extra_needed_left > 0: # Add the not-enough-space-on-the-right to the left
                new_x -= extra_needed_left
                new_width += extra_needed_left
        return new_x, new_width
    else:
        return x, width    
    

def crop_square(image_np, bbox, padding=0, divisible_by=8):
    """
    Crop a square region from the input image based on the bounding box with padding.

    Parameters:
    - image_np (numpy.ndarray): Input image as a numpy array.
    - bbox (list): Bounding box coordinates [x, y, width, height].
    - padding (int): Padding value (default is 0).
    - divisible_by (int): output width and height will be multiple of this

    Returns:
    - numpy.ndarray: Cropped image based on the new bounding box with padding.
    """
    image_height, image_width = image_np.shape[0], image_np.shape[1]

    # Get the new bounding box coordinates with padding
    x, y, width, height = get_bbox_with_padding(image_width, image_height, bbox, padding, divisible_by=divisible_by)

    # Perform the crop
    if len(image_np.shape) == 2:
        cropped_image = image_np[y:y + height, x:x + width]
    else:
        cropped_image = image_np[y:y + height, x:x + width, :]

    return cropped_image

def add_cropped_image_within_bbox(background, cropped_image_np, bbox, padding=0, divisible_by=8, do_resize=False):
    """
    Add a cropped image onto another image within the specified bounding box with padding.

    Parameters:
    - background (numpy.ndarray): Background image as a numpy array.
    - cropped_image_np (numpy.ndarray): Cropped image to be added.
    - bbox (list): Bounding box coordinates [x, y, width, height].
    - padding (int): Padding value.
    - divisible_by (int): output width and height will be multiple of this

    Returns:
    - numpy.ndarray: Resulting image with the cropped image added within the bounding box.
    """
    image_height, image_width = background.shape[0], background.shape[1]

    # Get the new bounding box coordinates with padding
    if padding != 0:
        x, y, width, height = get_bbox_with_padding(image_width, image_height, bbox, padding, divisible_by=8)
    else:
        x, y, width, height = bbox

    # Resize the cropped image to fit within the padded bounding box
    if do_resize:
        resized_cropped_image = Image.fromarray(cropped_image_np).resize((width, height), resample=Image.NEAREST)
        resized_cropped_image_np = np.array(resized_cropped_image)
    else:
        resized_cropped_image_np = cropped_image_np
    
    # Add the cropped image onto the background within the bounding box
    result_image = background.copy()
    result_image[y:y + height, x:x + width, :] = resized_cropped_image_np[:height, :width, :]

    return result_image

def add_segment_to_image(background, image, mask):
    """
    Add a segment from one image into another based on the provided mask.

    Parameters:
    - background (numpy.ndarray): Background image as a numpy array.
    - image (numpy.ndarray): Image segment to be added.
    - mask (numpy.ndarray): Mask indicating the region to be added from the image.

    Returns:
    - numpy.ndarray: Resulting image with the segment added.
    """
    result_image = background.copy()
    mask_binary = (mask > 0).astype(bool)
    result_image[mask_binary] = image[mask_binary]

    return result_image

def crop_bbox(image_np, bbox):
    """
    Crop a region from the input image based on the bounding box (not necessarily square) (without padding)

    Parameters:
    - image_np (numpy.ndarray): Input image as a numpy array.
    - bbox (list): Bounding box coordinates [x, y, width, height].

    Returns:
    - numpy.ndarray: Cropped image based on the new bounding box.
    """
    x, y, width, height = bbox
    x, y, width, height = int(x), int(y), int(width), int(height)
    # Perform the crop
    if len(image_np.shape) == 2:
        cropped_image = image_np[y:y + height, x:x + width]
    else:
        cropped_image = image_np[y:y + height, x:x + width, :]

    return cropped_image

# inspired by https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoDemo.ipynb

def load_coco_all(dataDir='coco', dataType='val2017'):
    coco = load_coco(dataDir='coco', dataType='val2017')
    coco_caps = load_coco_captions(dataDir='coco', dataType='val2017')
    return coco, coco_caps

def load_coco(dataDir='coco', dataType='val2017'):
    annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco = COCO(annFile)
    return coco

def get_categories():
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category['name'] for category in categories]
    return category_names

def get_images_from_categories(category_names):
    catIds = coco.getCatIds(catNms=category_names);
    imgIds = coco.getImgIds(catIds=catIds);
    return imgIds

def load_image(imgId):
    img = coco.loadImgs([imgId])[0] # We load only 1 image
    image_np = io.imread(img['coco_url'])
    return img, image_np

# https://www.flickr.com/services/api/misc.urls.html
# changing suffix z to b changes the resolution from 640 to 1024
def find_and_update_larger_image_on_flickr(imgId, do_update=True, bak_orig_dir=None):
    img = coco.loadImgs([imgId])[0] # We load only 1 image
    flickr_url = img['flickr_url']
    old_width, old_height = img['width'], img['height']
    larger_flickr_url = flickr_url.replace('_z.', '_b.')
    try:
        image_np = io.imread(larger_flickr_url)
    except Exception as e:
        print("Error while trying to read image with id %d from url %s" % (imgId, flickr_url))
        if bak_orig_dir:
            orig_path = "%s/%d_orig.png" % (bak_orig_dir, imgId)
            if os.path.exists(orig_path):
                print("Image found in backup orig dir!")
                image_np = np.array(Image.open(orig_path))
            else:
                raise e
        else:
            raise e
        
    new_width, new_height = image_np.shape[1], image_np.shape[0]
    #print("Aspect ratios:", (old_width / old_height), (new_width / new_height))
    if np.abs((old_width / old_height) - (new_width / new_height)) > 0.01: # Check if aspect ratios align (with 1% margin of error)
        print("Width and height swapped: rotate!") # TODO: this assumes it's always the same rotation though...
        #image_np = np.array(Image.fromarray(image_np).rotate(90))
        image_np = np.swapaxes(image_np, 0,1)[::-1,:,:]
        new_width, new_height = image_np.shape[1], image_np.shape[0]
        
    scale = new_width*1.0 / old_width
    #print(scale)

    if do_update:
        # Update img
        coco.imgs[imgId]['height'] = new_height
        coco.imgs[imgId]['width'] = new_width
        coco.imgs[imgId]['flickr_url'] = larger_flickr_url
        img = coco.loadImgs([imgId])[0] 
        
        # Update annotations
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
        # iscrowd=True has another way of representing segms, which is not compatible with my update
        for annId in annIds:
            annotation = coco.anns[annId]
            coco.anns[annId] = scale_annotation(annotation, scale)
    return img, image_np, old_width, old_height, new_width, new_height, scale

def scale_annotation(annotation, scale):
    segm = annotation['segmentation']
    if isinstance(segm, list):
        annotation['segmentation'] =  [ [y*scale for y in x] for x in segm]
    elif isinstance(segm, dict) and 'counts' in segm.keys():
        annotation['segmentation']['counts'] = [ [y*scale for y in x] for x in segm['counts']]
    else:
        print("Can't handle segm of ann %d" % annotation['id'])
        
    annotation['bbox'] = [ a*scale for a in annotation['bbox']]
    annotation['area'] = scale * scale * annotation['area']
    return annotation

def get_annotations(category_names, img):
    catIds = coco.getCatIds(catNms=category_names);
    annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=False)
    annotations = coco.loadAnns(annIds)
    return annotations

def show_category_mask(image_np, category_names, img):
    plt.imshow(image_np); plt.axis('off')
    anns = get_annotations(category_names, img)
    coco.showAnns(anns)

def load_coco_captions(dataDir='coco', dataType='val2017'):
    captionsFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
    coco_caps = COCO(captionsFile)
    return coco_caps

def get_captions_from_image(img):
    captionsIds = coco_caps.getAnnIds(imgIds=img['id']);
    captions = coco_caps.loadAnns(captionsIds)
    return captions

def replace_category_in_prompt(prompt, category_old, category_new, give_warning=False):
    new_prompt = prompt.replace(category_old, category_new)
    #assert prompt != new_prompt, "No words replaced in prompt %s (%s -> %s)" % (prompt, category_old, category_new)
    if give_warning and prompt == new_prompt:
        print("WARNING! No words replaced in prompt %s (%s -> %s)" % (prompt, category_old, category_new))
    return new_prompt

class SaveIntermediateImage(Callable):
    def __init__(self, generator, save_dir=None, save_filename=""):
        self.generator = generator
        self.save_dir = save_dir
        self.save_filename = save_filename
    
    def __call__(self, pipeline: 'DiffusionPipeline', step: int, timestep: int, callback_kwargs: Dict):
        #print(f"Step end callback called at step {step} with timestep {timestep}")
        #print(f"Callback keys:")
        #print(list(callback_kwargs.keys()))
        #print(f"Callback kwargs: {callback_kwargs}")

        if 'latents' in callback_kwargs.keys():
            latents = callback_kwargs['latents'] #64x64x3
            # see https://github.com/huggingface/diffusers/blob/76696dca558267999abf3e7c29e1a256cbcb407a/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py#L1435
            intermediate_images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False, generator=self.generator)[0] # 512x512x3
            
            do_denormalize = [True] * latents.shape[0]
            intermediate_images = pipe.image_processor.postprocess(intermediate_images, do_denormalize=do_denormalize)
            intermediate_image = intermediate_images[0] # TODO: support for multiple images
    
            if save_dir is not None:
                intermediate_image_filename = "%s/%s_step%d.png" % (save_dir, save_filename, step)
                intermediate_image.save(intermediate_image_filename)
                print(f"Saved intermediate image to %s" % intermediate_image_filename)
        else:
            print("No latents at step %d" % step)
        
        # Save intermediate image?
        return callback_kwargs

# generator: can define with manual seed, to make deterministic
def inpaint_with_prompt(pipe, prompt, image_np, mask_np, negative_prompt="",
                        width=None, height=None,
                        num_inference_steps=50, num_images_per_prompt=1, padding_mask_crop=None, 
                        guidance_scale=7.5, strength=1.0,
                        generator=None, seed=None, callback_on_step_end=None):
    # for options, see: https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/inpaint

    if generator is None:
        if seed is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            generator = torch.Generator(device=device)
            generator.manual_seed(seed)
    
    generated_images_output = pipe(prompt=prompt, negative_prompt=negative_prompt,
                            width=width, height=height,
                            image=Image.fromarray(image_np), mask_image=Image.fromarray(mask_np),
                           num_inference_steps=num_inference_steps, num_images_per_prompt=num_images_per_prompt,
                           padding_mask_crop=padding_mask_crop,
                            guidance_scale=guidance_scale, strength=strength,
                           generator=generator, callback_on_step_end = callback_on_step_end)
    # output_type=np.array -> should be PIL when using padding_mask_crop
    generated_images = generated_images_output.images
    if hasattr(generated_images_output, 'nsfw_content_detected'):
        nsfw_detected = generated_images_output.nsfw_content_detected
    else:
        nsfw_detected = None
    return generated_images, nsfw_detected


def find_caption_with_word(captions, word):
    """
    Find the first caption that contains the specified word.

    Parameters:
    - captions (list): List of dictionaries containing captions.
    - word (str): Word to search for in the captions.

    Returns:
    - str or None: The first caption containing the word, or first caption if word was not found.
    """
    for caption_dict in captions:
        caption = caption_dict['caption']
        if word in caption:
            return caption
    return captions[0]['caption'] # default: return first caption

def itm(image, question, print_score=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = blip_processor(image, question, return_tensors="pt").to(device, torch.float16)
    itm_scores = blip_itm_model(**inputs)[0]
    itm_score = torch.nn.functional.softmax(itm_scores, dim=1)[:,1].item()
    cos_score = blip_itm_model(**inputs, use_itm_head=False)[0].item() # cosine score
    if print_score:
        print("ITM: %.2f" % itm_score)
        print("Cos: %.2f" % cosine_score)

    return itm_score, cos_score


# Do inpainting for several images at once
# verbose = 0: minimum output, 1: only show results, 2: + show original, 3: + show mask, 4: + show crop
def do_inpainting_loop(inpainting_model_name, category, category_new,
                       use_prompt_from_caption=False, use_negative_prompt=0,
                       max_images=None,
                       verbose=0, save_dir="", save_orig_dir="", bak_orig_dir="",
                       use_bbox_mask=True, use_segm_mask=True,
                       use_all_masks=False, # TODO: enable to iteratively have more masks?
                       use_ps_mask_for_splicing=True, ps_mask_dir=None,
                       num_inference_steps_min=10, num_inference_steps_max=50,
                       guidance_scale_min=1, guidance_scale_max=10,
                       strength=1.0,
                       num_images_per_prompt=3,
                       seed_num_inference_steps=26, seed_guidance_scale=9, seed_strength=1993, seed_gen=26091993, fix_seed_gen=False,
                       do_nima=False, do_itm=False, do_nima_on_full=False, do_giqa=False, do_giqa_on_full=False,
                       skip_if_exists=False, skip_dir="",
                       stop_on_exception=False):   

    REQUIRED_IMAGE_WIDTH = 512
    REQUIRED_IMAGE_HEIGHT = 512
    MIN_MASK_SIZE = 64*64

    random_num_inference_steps = np.random.default_rng(seed=seed_num_inference_steps)
    random_guidance_scale = np.random.default_rng(seed=seed_guidance_scale)
    #random_strength = np.random.default_rng(seed=seed_strength)
    if not fix_seed_gen:
        random_gen = np.random.default_rng(seed=seed_gen)

    # These IDs became available later, and mess up the first-50-images thing
    imgIdsBlacklistDict = { "toilet": [201775] }
    
    category_names = [category]
    imgIds = get_images_from_categories(category_names)
    print("Found %d images with category %s" % (len(imgIds), category))
    print(imgIds)
    images_to_inpaint = len(imgIds) if max_images is None else min(max_images, len(imgIds))
    print("Will inpaint first %d images" % images_to_inpaint)

    if save_orig_dir:
        ensure_dir_from_dir(save_orig_dir)
        csv_path = "%s/metadata.csv" % save_orig_dir
        csvfile = open(csv_path, 'w', newline='')
        writer = csv.writer(csvfile)
        header = ['image_id', 'width', 'height', 'prompt', 'bbox_512', 'num_inference_steps', 'guidance_scale', 'seed']
        if do_nima:
            header.extend(['orig_nima_mean', 'orig_nima_std'])
        if do_giqa:
            header.extend(['orig_giqa'])
        if do_itm:
            header.extend(['orig_itm', 'orig_cos'])
        writer.writerow(header)

    if save_dir:
        ensure_dir_from_dir(save_dir)
        csv_path_analysis = "%s/metadata_analysis.csv" % save_dir
        csvfile_analysis = open(csv_path_analysis, 'w', newline='')
        writer_analysis = csv.writer(csvfile_analysis)
        header_analysis = ['filename']
        if do_nima:
            header_analysis.extend(['nima_mean', 'nima_std'])
        if do_giqa:
            header_analysis.extend(['giqa_score'])
        if do_itm:
            header_analysis.extend(['itm', 'cos'])
        #header_analysis.append("NSFW detected")
        writer_analysis.writerow(header_analysis)

    inpainted_images = 0
    for i in range(len(imgIds)):
        if inpainted_images >= images_to_inpaint:
            break
        try:
            # Load image
            ####
            imgId = imgIds[i]
            print("Image %d with id %d at %s" % (i, imgId, datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
            if category in imgIdsBlacklistDict:
                assert imgId not in imgIdsBlacklistDict[category], "Image ID in blacklist"
                
            img, image_np, old_width, old_height, img_width, img_height, scale = find_and_update_larger_image_on_flickr(imgId, do_update=True, bak_orig_dir=bak_orig_dir)

            assert img_width > REQUIRED_IMAGE_WIDTH and img_height >= REQUIRED_IMAGE_HEIGHT, "Image resolution (%dx%d) too small for model" % (img_width, img_height)

            # Mask
            ####
            annotations = get_annotations(category_names, img)
            annotation_i = 0
            annotations_to_use = annotations if use_all_masks else annotations[annotation_i:annotation_i+1]

            main_mask_bbox = merge_bboxes([ann['bbox'] for ann in annotations_to_use]) # Main bbox is the same for both bbox and segm mask
            _, _, main_mask_bbox_width, main_mask_bbox_height = main_mask_bbox
            mask_size = main_mask_bbox_width*main_mask_bbox_height
            #print("Mask size (%dx%d) " % (main_mask_bbox_width, main_mask_bbox_height))
            assert main_mask_bbox_width < REQUIRED_IMAGE_WIDTH and main_mask_bbox_height < REQUIRED_IMAGE_HEIGHT, "Mask resolution (%dx%d) too large for model" % (main_mask_bbox_width, main_mask_bbox_height)
            assert mask_size > MIN_MASK_SIZE, "Mask resolution (%dx%d) too small" % (main_mask_bbox_width, main_mask_bbox_height)

            # Seed
            if not fix_seed_gen:
                seed = int(random_gen.integers(low=0, high=np.iinfo(np.int32).max, dtype=np.int32))

            num_inference_steps = int(random_num_inference_steps.integers(low=num_inference_steps_min, high=num_inference_steps_max+1, dtype=np.int32))
            guidance_scale = guidance_scale_min + (random_guidance_scale.random() * (guidance_scale_max - guidance_scale_min))
            #strength = strength_min + (random_strength.random() * (strength_max - strength_min))
            
            mask_bbox_np = None
            mask_segm_np = None
            try_again = True
            while .:
                for annotation in annotations_to_use:
                    bbox = annotation['bbox']
                    mask_np_single = create_mask_from_bbox(bbox, image_size=(img_width, img_height))
                    mask_bbox_np = mask_np_single if mask_bbox_np is None else np.maximum(mask_bbox_np, mask_np_single)
                    mask_np_single = coco.annToMask(annotation) * 255
                    mask_segm_np = mask_np_single if mask_segm_np is None else np.maximum(mask_segm_np, mask_np_single)
                    
                if np.amax(mask_bbox_np) > 0 or np.amax(mask_segm_np) > 0:
                    try_again = False
                elif not use_all_masks and annotation_i < len(annotations):
                    annotation_i += 1
                    annotations_to_use = annotations[annotation_i:annotation_i+1]
                else:
                    assert np.amax(mask_bbox_np) > 0 and np.amax(mask_segm_np) > 0, "All bbox masks do not select anything"

            # Prompt
            ###
            negative_prompt = category if use_negative_prompt and category != category_new else ""
            if use_prompt_from_caption == 1 or use_prompt_from_caption == 2:
                captions = get_captions_from_image(img)
                caption = find_caption_with_word(captions, category)
                prompt = replace_category_in_prompt(caption, category, category_new, give_warning=category != category_new)
                # TODO: filter out newlines from caption
                if use_prompt_from_caption == 2:
                    prompt = "%s, %s" % (category_new, prompt)
            else:
                prompt = category_new
            
            if verbose >= 1:
                print("Prompt: %s" % prompt)

            if inpainting_model_name == "sd2":
                # 512x512 crop
                crop_size = 512
                crop_width, crop_height = REQUIRED_IMAGE_WIDTH, REQUIRED_IMAGE_HEIGHT
            else:
                # max 1024 crop, but has to be divisable by 8!
                crop_size = 1024
                crop_width, crop_height = image_np.shape[1] // 8 * 8, image_np.shape[0] // 8 * 8

            # Get 512x512 crop around mask
            ###
            image_np_512, bbox_512 = create_and_crop_larger_bbox_around_bbox(image_np, main_mask_bbox, crop_width, crop_height)
            mask_np_512 = create_mask_from_bbox(bbox_512, image_size=(img_width, img_height))
            if use_bbox_mask:
                mask_bbox_np_512, _ = create_and_crop_larger_bbox_around_bbox(mask_bbox_np, main_mask_bbox, crop_width, crop_height)
            if use_segm_mask:
                mask_segm_np_512, _ = create_and_crop_larger_bbox_around_bbox(mask_segm_np, main_mask_bbox, crop_width, crop_height)
                
            # Generate images
            ###
            skipped_bbox_gen = False
            if use_bbox_mask:
                g_i = 0
                if inpainting_model_name == "ps":
                    gen_path = "%s/%d_mask_bbox.png_%s_%d.png" % (skip_dir, imgId, inpainting_model_name, 0) # Check if first one exists
                else:
                    gen_path = "%s/%d_mask_bbox.png_%s-%d_%d.png" % (skip_dir, imgId, inpainting_model_name, crop_size, g_i) # Check if first one exists
                if skip_if_exists and os.path.exists(gen_path):
                    skipped_bbox_gen = True
                    print("ImageId %d bbox already found" % imgId)
                    generated_image_512_bbox_np_list = []
                    generated_image_combined_bbox_np_list = []
                    for g_i in range(num_images_per_prompt):
                        if inpainting_model_name == "ps":
                            gen_path = "%s/%d_mask_bbox.png_%s_%d.png" % (skip_dir, imgId, inpainting_model_name, g_i)
                            generated_image_combined_bbox_np = np.array(Image.open(gen_path))
                            generated_image_combined_bbox_np_list.append(generated_image_combined_bbox_np)
                            # Crop 512
                            generated_image_512_bbox_np, bbox_512 = create_and_crop_larger_bbox_around_bbox(generated_image_combined_bbox_np, main_mask_bbox, crop_width, crop_height)
                            generated_image_512_bbox_np_list.append(generated_image_512_bbox_np)
                        else:
                            # Read 512
                            gen_path = "%s/%d_mask_bbox.png_%s-%d_%d.png" % (skip_dir, imgId, inpainting_model_name, crop_size, g_i)
                            generated_image_512_bbox_np = np.array(Image.open(gen_path))
                            generated_image_512_bbox_np_list.append(generated_image_512_bbox_np)
                    generated_image_512_bbox_nsfw_list = None
                
                else:
                    generated_image_512_bbox_pil_list, generated_image_512_bbox_nsfw_list = inpaint_with_prompt(pipe, prompt, image_np_512, mask_bbox_np_512,
                                                                    width=image_np_512.shape[1], height=image_np_512.shape[0],
                                                                    num_inference_steps=num_inference_steps, num_images_per_prompt=num_images_per_prompt,
                                                                    guidance_scale=guidance_scale, strength=strength,
                                                                    negative_prompt=negative_prompt, seed=seed)
                    generated_image_combined_bbox_np_list = []
                    #generated_image_512_bbox_pil_list = generated_images_512_bbox.images
                    #generated_image_512_bbox_nsfw_list = generated_images_512_bbox.nsfw_content_detected
                    generated_image_512_bbox_np_list = []
                    
                    for generated_image_512_bbox_pil in generated_image_512_bbox_pil_list:
                        generated_image_512_bbox_np = np.array(generated_image_512_bbox_pil)
                        generated_image_512_bbox_np_list.append(generated_image_512_bbox_np)
                        generated_image_combined_bbox_np = add_cropped_image_within_bbox(image_np, generated_image_512_bbox_np, bbox_512, padding=0)
                        generated_image_combined_bbox_np_list.append(generated_image_combined_bbox_np)

                    if use_ps_mask_for_splicing:
                        mask_ps_path = "%s/%d_mask_bbox.png_ps_mask.png" % (ps_mask_dir, imgId)
                        if os.path.exists(mask_ps_path):
                            ps_mask_bbox_found = True
                            mask_ps_pil = Image.open(mask_ps_path).convert("L") # 8 bit
                            generated_image_combined_bbox_ps_pil_list = []
                            for g_i, generated_image_combined_bbox_np in enumerate(generated_image_combined_bbox_np_list):
                                generated_image_combined_bbox_ps_pil = Image.composite(Image.fromarray(generated_image_combined_bbox_np), Image.fromarray(image_np), mask_ps_pil)
                                generated_image_combined_bbox_ps_pil_list.append(generated_image_combined_bbox_ps_pil)
                        else:
                            print("No PS mask bbox found: %s" % mask_ps_path)
                            ps_mask_bbox_found = False
                    
            skipped_segm_gen = False
            if use_segm_mask:
                g_i = 0
                if inpainting_model_name == "ps":
                    gen_path = "%s/%d_mask_segm.png_%s_%d.png" % (skip_dir, imgId, inpainting_model_name, g_i)
                else:
                    gen_path = "%s/%d_mask_segm.png_%s-%d_%d.png" % (skip_dir, imgId, inpainting_model_name, crop_size, g_i)
                if skip_if_exists and os.path.exists(gen_path):
                    skipped_segm_gen = True
                    print("ImageId %d segm already found" % imgId)
                    generated_image_512_segm_np_list = []
                    generated_image_combined_segm_np_list = []
                    for g_i in range(num_images_per_prompt):
                        if inpainting_model_name == "ps":
                            gen_path = "%s/%d_mask_segm.png_%s_%d.png" % (skip_dir, imgId, inpainting_model_name, g_i)
                            generated_image_combined_segm_np = np.array(Image.open(gen_path))
                            generated_image_combined_segm_np_list.append(generated_image_combined_segm_np)
                            # Crop 512
                            generated_image_512_segm_np, bbox_512 = create_and_crop_larger_bbox_around_bbox(generated_image_combined_segm_np, main_mask_bbox, crop_width, crop_height)
                            generated_image_512_segm_np_list.append(generated_image_512_segm_np)
                        else:
                            # Read 512
                            gen_path = "%s/%d_mask_segm.png_%s-%d_%d.png" % (skip_dir, imgId, inpainting_model_name, crop_size, g_i)
                            generated_image_512_segm_np = np.array(Image.open(gen_path))
                            generated_image_512_segm_np_list.append(generated_image_512_segm_np)
                    generated_image_512_segm_nsfw_list = None
                else:
                    generated_image_512_segm_pil_list, generated_image_512_segm_nsfw_list = inpaint_with_prompt(pipe, prompt, image_np_512, mask_segm_np_512,
                                                                    width=image_np_512.shape[1], height=image_np_512.shape[0],
                                                                    num_inference_steps=num_inference_steps, num_images_per_prompt=num_images_per_prompt,
                                                                    guidance_scale=guidance_scale, strength=strength,
                                                                    negative_prompt=negative_prompt, seed=seed)
                    generated_image_combined_segm_np_list = []
                    generated_image_512_segm_np_list = []
                    for generated_image_512_segm_pil in generated_image_512_segm_pil_list:
                        generated_image_512_segm_np = np.array(generated_image_512_segm_pil)
                        generated_image_512_segm_np_list.append(generated_image_512_segm_np)
                        generated_image_combined_segm_np = add_cropped_image_within_bbox(image_np, generated_image_512_segm_np, bbox_512, padding=0)
                        generated_image_combined_segm_np_list.append(generated_image_combined_segm_np)
                        
                    if use_ps_mask_for_splicing:
                        mask_ps_path = "%s/%d_mask_segm.png_ps_mask.png" % (ps_mask_dir, imgId)
                        if os.path.exists(mask_ps_path):
                            ps_mask_segm_found = True
                            mask_ps_pil = Image.open(mask_ps_path).convert("L") # 8 bit
                            generated_image_combined_segm_ps_pil_list = []
                            for g_i, generated_image_combined_segm_np in enumerate(generated_image_combined_segm_np_list):
                                generated_image_combined_segm_ps_pil = Image.composite(Image.fromarray(generated_image_combined_segm_np), Image.fromarray(image_np), mask_ps_pil)
                                generated_image_combined_segm_ps_pil_list.append(generated_image_combined_segm_ps_pil)
                        else:
                            ps_mask_segm_found = False

            #if skipped_bbox_gen or skipped_segm_gen:
            #    inpainted_images += 1
            #    continue
                
            if do_nima or do_itm or do_giqa:
                main_mask_bbox_512 = main_mask_bbox[0] - bbox_512[0], main_mask_bbox[1] - bbox_512[1], main_mask_bbox[2], main_mask_bbox[3]
                image_np_512_only_bbox = crop_bbox(image_np_512, main_mask_bbox_512)
                if do_nima:
                    # Original NIMA
                    if save_orig_dir:
                        if do_nima_on_full:
                            nima_mean_orig, nima_std_orig = nima.assess_quality(image_np_512, nima_model)
                        else:
                            nima_mean_orig, nima_std_orig = nima.assess_quality(image_np_512_only_bbox, nima_model)
                
                    # Fake NIMA
                    if use_bbox_mask:
                        nima_means_bbox = np.zeros(num_images_per_prompt)
                        nima_stds_bbox = np.zeros(num_images_per_prompt)
                        for g_i, generated_image_512_bbox_np in enumerate(generated_image_512_bbox_np_list):
                            if do_nima_on_full:
                                nima_means_bbox[g_i], nima_stds_bbox[g_i] = nima.assess_quality(np.array(generated_image_512_bbox_np), nima_model)
                            else:
                                generated_image_512_bbox_pil_only_bbox = crop_bbox(generated_image_512_bbox_np, main_mask_bbox_512)
                                nima_means_bbox[g_i], nima_stds_bbox[g_i] = nima.assess_quality(np.array(generated_image_512_bbox_pil_only_bbox), nima_model)
                    if use_segm_mask:
                        nima_means_segm = np.zeros(num_images_per_prompt)
                        nima_stds_segm = np.zeros(num_images_per_prompt)
                        for g_i, generated_image_512_segm_np in enumerate(generated_image_512_segm_np_list):
                            if do_nima_on_full:
                                nima_means_segm[g_i], nima_stds_segm[g_i] = nima.assess_quality(np.array(generated_image_512_segm_np), nima_model)
                            else:
                                generated_image_512_segm_pil_only_bbox = crop_bbox(generated_image_512_segm_np, main_mask_bbox_512)
                                nima_means_segm[g_i], nima_stds_segm[g_i] = nima.assess_quality(np.array(generated_image_512_segm_pil_only_bbox), nima_model)
                if do_giqa:
                    # Original NIMA
                    if save_orig_dir:
                        if do_giqa_on_full:
                            giq_score_orig = gmm_score.get_activation_preloaded_model(image_np_512, giqa_model, pca=giqa_pca, pca_gmm=giqa_pca_gmm, read_files=False)
                        else:
                            giqa_score_orig = gmm_score.get_activation_preloaded_model(image_np_512_only_bbox, giqa_model, pca=giqa_pca, pca_gmm=giqa_pca_gmm, read_files=False)
                
                    # Fake NIMA
                    if use_bbox_mask:
                        giqa_scores_bbox = np.zeros(num_images_per_prompt)
                        for g_i, generated_image_512_bbox_np in enumerate(generated_image_512_bbox_np_list):
                            if do_giqa_on_full:
                                giqa_scores_bbox[g_i] = gmm_score.get_activation_preloaded_model(np.array(generated_image_512_bbox_np), giqa_model, pca=giqa_pca, pca_gmm=giqa_pca_gmm, read_files=False)
                            else:
                                generated_image_512_bbox_pil_only_bbox = crop_bbox(generated_image_512_bbox_np, main_mask_bbox_512)
                                giqa_scores_bbox[g_i] = gmm_score.get_activation_preloaded_model(np.array(generated_image_512_bbox_pil_only_bbox), giqa_model, pca=giqa_pca, pca_gmm=giqa_pca_gmm, read_files=False)
                    if use_segm_mask:
                        giqa_scores_segm = np.zeros(num_images_per_prompt)
                        for g_i, generated_image_512_segm_np in enumerate(generated_image_512_segm_np_list):
                            if do_giqa_on_full:
                                giqa_scores_segm[g_i] = gmm_score.get_activation_preloaded_model(np.array(generated_image_512_segm_np), giqa_model, pca=giqa_pca, pca_gmm=giqa_pca_gmm, read_files=False)
                            else:
                                generated_image_512_segm_pil_only_bbox = crop_bbox(generated_image_512_segm_np, main_mask_bbox_512)
                                giqa_scores_segm[g_i] = gmm_score.get_activation_preloaded_model(np.array(generated_image_512_segm_pil_only_bbox), giqa_model, pca=giqa_pca, pca_gmm=giqa_pca_gmm, read_files=False)
                                
                if do_itm:
                    # Original
                    if save_orig_dir:
                        itm_question = category
                        itm_score_orig, itm_cos_orig = itm(image_np_512_only_bbox, itm_question)

                    itm_question = category_new
                    if use_bbox_mask:
                        itm_scores_bbox = np.zeros(num_images_per_prompt)
                        itm_cosines_bbox = np.zeros(num_images_per_prompt)
                        for g_i, generated_image_512_bbox_np in enumerate(generated_image_512_bbox_np_list):
                            generated_image_512_bbox_pil_only_bbox = crop_bbox(generated_image_512_bbox_np, main_mask_bbox_512)
                            itm_scores_bbox[g_i], itm_cosines_bbox[g_i] = itm(np.array(generated_image_512_bbox_pil_only_bbox), itm_question)
                    if use_segm_mask:
                        itm_scores_segm = np.zeros(num_images_per_prompt)
                        itm_cosines_segm = np.zeros(num_images_per_prompt)
                        for g_i, generated_image_512_segm_np in enumerate(generated_image_512_segm_np_list):
                            generated_image_512_segm_pil_only_bbox = crop_bbox(generated_image_512_segm_np, main_mask_bbox_512)
                            itm_scores_segm[g_i], itm_cosines_segm[g_i] = itm(np.array(generated_image_512_segm_pil_only_bbox), itm_question)
                    
            # Save images
            ###
            if save_orig_dir:
                # Save metadata
                #header =      ['imgId', 'width', 'height', 'prompt', 'bbox_512', 'num_inference_steps', 'seed']
                row = [imgId, img_width, img_height, prompt, bbox_512, num_inference_steps, guidance_scale, seed]
                if do_nima:
                    row.extend([nima_mean_orig, nima_std_orig])
                if do_giqa:
                    row.extend([giqa_score_orig])
                if do_itm:
                    row.extend([itm_score_orig, itm_cos_orig])
                writer.writerow(row)

                # Save images
                if not (skipped_bbox_gen or skipped_segm_gen):
                    Image.fromarray(image_np).save("%s/%d_orig.png" % (save_orig_dir, imgId))
                    if use_bbox_mask:
                        Image.fromarray(mask_bbox_np).save("%s/%d_mask_bbox.png" % (save_orig_dir, imgId))
                    if use_segm_mask:
                        Image.fromarray(mask_segm_np).save("%s/%d_mask_segm.png" % (save_orig_dir, imgId))
                        
                    Image.fromarray(image_np_512).save("%s/%d_orig_%d.png" % (save_orig_dir, imgId, crop_size))
                    Image.fromarray(mask_np_512).save("%s/%d_mask_%d.png" % (save_orig_dir, imgId, crop_size))
            
                    if use_bbox_mask and not skipped_bbox_gen:
                        Image.fromarray(mask_bbox_np_512).save("%s/%d_mask_bbox_%d.png" % (save_orig_dir, imgId, crop_size))
                    if use_segm_mask and not skipped_segm_gen:
                        Image.fromarray(mask_segm_np_512).save("%s/%d_mask_segm_%s.png" % (save_orig_dir, imgId, crop_size))
                    
            if save_dir:
                if use_bbox_mask:
                    for g_i, (generated_image_512_bbox_np) in enumerate(generated_image_512_bbox_np_list):
                        if inpainting_model_name == "ps":
                            gen_512_name = "%d_mask_bbox.png_%s_%d.png" % (imgId, inpainting_model_name, g_i)
                        else:
                            gen_512_name = "%d_mask_bbox.png_%s-%d_%d.png" % (imgId, inpainting_model_name, crop_size, g_i)
                        if not skipped_bbox_gen:
                            generated_image_combined_bbox_np = generated_image_combined_bbox_np_list[g_i]
                            gen_512_path = "%s/%s" % (save_dir, gen_512_name)
                            if not skipped_bbox_gen:
                                Image.fromarray(generated_image_512_bbox_np).save(gen_512_path)
                                if inpainting_model_name == "sd2": # Don't have combined version with 512-splice
                                    Image.fromarray(generated_image_combined_bbox_np).save("%s/%d_mask_bbox.png_%s_%d.png" % (save_dir, imgId, inpainting_model_name, g_i))
                    
                        row_analysis = [gen_512_name]
                        if do_nima:
                            row_analysis.extend([nima_means_bbox[g_i], nima_stds_bbox[g_i]])
                        if do_giqa:
                            row_analysis.extend([giqa_scores_bbox[g_i]])
                        if do_itm:
                            row_analysis.extend([itm_scores_bbox[g_i], itm_cosines_bbox[g_i]])
                        nsfw_detected = generated_image_512_bbox_nsfw_list[g_i] if generated_image_512_bbox_nsfw_list and len(generated_image_512_bbox_nsfw_list) > 0 else "UNKNOWN"
                        #row_analysis.append(nsfw_detected)
                        writer_analysis.writerow(row_analysis)
                            
                        if not skipped_bbox_gen and use_ps_mask_for_splicing and ps_mask_bbox_found:
                            generated_image_combined_bbox_ps_pil_list[g_i].save("%s/%d_mask_bbox.png_ps_mask.png_%s_%d.png" % (save_dir, imgId, inpainting_model_name, g_i))
                                
                if use_segm_mask:
                    for g_i, generated_image_512_segm_np in enumerate(generated_image_512_segm_np_list):
                        if inpainting_model_name == "ps":
                            gen_512_name = "%d_mask_segm.png_%s_%d.png" % (imgId, inpainting_model_name, g_i)
                        else:
                            gen_512_name = "%d_mask_segm.png_%s-%d_%d.png" % (imgId, inpainting_model_name, crop_size, g_i)
                        if not skipped_segm_gen:
                            generated_image_combined_segm_np = generated_image_combined_segm_np_list[g_i]
                            gen_512_path = "%s/%s" % (save_dir, gen_512_name)
                            if not skipped_segm_gen:
                                Image.fromarray(generated_image_512_segm_np).save(gen_512_path)
                                
                                if inpainting_model_name == "sd2": # Don't have combined version with 512-splice for sdxl
                                    Image.fromarray(generated_image_combined_segm_np).save("%s/%d_mask_segm.png_%s_%d.png" % (save_dir, imgId, inpainting_model_name, g_i))
                    
                        row_analysis = [gen_512_name]
                        if do_nima:
                            row_analysis.extend([nima_means_segm[g_i], nima_stds_segm[g_i]])
                        if do_giqa:
                            row_analysis.extend([giqa_scores_segm[g_i]])
                        if do_itm:
                            row_analysis.extend([itm_scores_segm[g_i], itm_cosines_segm[g_i]])
                        nsfw_detected = generated_image_512_segm_nsfw_list[g_i] if generated_image_512_segm_nsfw_list and len(generated_image_512_segm_nsfw_list) > 0 else "UNKNOWN"
                        #row_analysis.append(nsfw_detected)
                        writer_analysis.writerow(row_analysis)
                            
                        if not skipped_segm_gen and use_ps_mask_for_splicing and ps_mask_segm_found:
                            generated_image_combined_segm_ps_pil_list[g_i].save("%s/%d_mask_segm.png_ps_mask.png_%s_%d.png" % (save_dir, imgId, inpainting_model_name, g_i))
                                
            inpainted_images += 1
            print()
        except HTTPError as e:
            #print("HTTPError for image id %d" % imgId)
            print()
            continue
        except AssertionError as e:
            #print("AssertionError for image id %d" % imgId)
            print(e)
            print()
            continue
        except KeyboardInterrupt as e:
            print("Stopping in style because of KeyboardInterrupt")
            if save_orig_dir:
                csvfile.close()
            if save_dir:
                csvfile_analysis.close()
            raise e
        except Exception as e:
            if stop_on_exception:
                if save_orig_dir:
                    csvfile.close()
                if save_dir:
                    csvfile_analysis.close()
                raise e
            else:
                print("Exception!")
                traceback.print_exc()
                print("Continuing...")
                print()
                continue   
    if save_orig_dir:
        csvfile.close()
    if save_dir:
        csvfile_analysis.close()


# Global variables
pipe = None
def load_pipeline(model_name, cache_dir="./checkpoints"):
    if model_name == "sd2":
        model_name_full = "stabilityai/stable-diffusion-2-inpainting"
    elif model_name == "sdxl":
        model_name_full = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    else:
        print("Model %s not supported" % model_name)
        return None
    global pipe
    pipe = AutoPipelineForInpainting.from_pretrained(
    #pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_name_full,
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    return pipe

# Global variables
coco = None
coco_caps = None
def load_coco_all(dataDir='coco', dataType='val2017'):
    global coco, coco_caps
    coco = load_coco(dataDir='coco', dataType='val2017')
    coco_caps = load_coco_captions(dataDir='coco', dataType='val2017')
    return coco, coco_caps

nima_model = None
def load_nima(nima_model_path = "./NIMA/checkpoints/epoch-82.pth", vgg16_model_path = "./NIMA/checkpoints/vgg16-397923af.pth"):
    global nima_model
    nima_model = nima.load_model(nima_model_path, vgg16_model_path=vgg16_model_path)
    return nima_model

giqa_model, giqa_pca, giqa_pca_gmm = None, None, None
def load_giqa(pca_path = "./GIQA/checkpoints/pca95-cat.pkl", pca_gmm_path = "./GIQA/checkpoints/gmm-cat-pca95-full7.pkl", inception_dims=2048):
    global giqa_model, giqa_pca, giqa_pca_gmm
    useCuda = torch.cuda.is_available()
    giqa_model, giqa_pca, giqa_pca_gmm = gmm_score.load_models(inception_dims, useCuda, pca_path, pca_gmm_path)
    return giqa_model, giqa_pca, giqa_pca_gmm

# For Image-Text Matching
blip_processor, blip_itm_model = None, None
def load_blip_itm(cache_dir="./checkpoints"):
    global blip_processor, blip_itm_model
    blip_processor = BlipProcessor.from_pretrained("%s/Salesforce_blip-itm-base-coco" % cache_dir)
    blip_itm_model = BlipForImageTextRetrieval.from_pretrained("%s/Salesforce_blip-itm-base-coco" % cache_dir, torch_dtype=torch.float16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blip_itm_model.to(device)
    
    return blip_processor, blip_itm_model

def main():
    parser = argparse.ArgumentParser(description='Your description here')
    parser.add_argument('--inpainting_model_name', type=str, default='sd2')
    parser.add_argument('--cache_dir', type=str, default='./checkpoints')
    parser.add_argument('--coco_data_dir', type=str, default='coco')
    parser.add_argument('--coco_data_type', type=str, default='val2017')
    parser.add_argument('--nima_model_path', type=str, default='./NIMA/checkpoints/epoch-82.pth')
    parser.add_argument('--vgg16_model_path', type=str, default='./NIMA/checkpoints/vgg16-397923af.pth')
    parser.add_argument('--giqa_pca_path', type=str, default="./GIQA/checkpoints/pca95-cat.pkl")
    parser.add_argument('--giqa_pca_gmm_path', type=str, default="./GIQA/checkpoints/gmm-cat-pca95-full7.pkl")
    parser.add_argument('--blip_processor_path', type=str, default="./checkpoints/Salesforce_blip-itm-base-coco")
    parser.add_argument('--blip_itm_model_path', type=str, default="./checkpoints/Salesforce_blip-itm-base-coco")
    
    parser.add_argument('--save_dir', type=str, default='output', help='Directory to save all generated files in (with category as subdirectory)')
    parser.add_argument('--save_orig_dir', type=str, default='', help='Directory to save all original and mask files in (with category as subdirectory)')
    parser.add_argument('--bak_orig_dir', type=str, default='', help='Directory where save original files were previously saved, to be used as backup in case we cannot retrieve the flickr url in (with category as subdirectory)')
    parser.add_argument('--categories', type=str, default='dog', help='Category to change (comma separated)')
    parser.add_argument('--categories_new', type=str, default='dog', help='New category (comma separated)')
    parser.add_argument('--max_images', type=int, default=None, help='Maximum number of images')
    parser.add_argument('--use_bbox_mask', action='store_true', help='Use bounding box mask')
    parser.add_argument('--use_segm_mask', action='store_true', help='Use segm mask')
    parser.add_argument('--use_all_masks', action='store_true', help='Use all masks')
    parser.add_argument('--use_ps_mask_for_splicing', action='store_true', help='Use Photoshop mask for masking mask')
    parser.add_argument('--ps_mask_dir', type=str, help='Directory in which photoshop masks are given')
    parser.add_argument('--use_prompt_from_caption', type=int, default=1, help='Use prompt from caption (0 use caption, 1 use caption, 2 use both category and caption)')
    parser.add_argument('--use_negative_prompt', action='store_true', help='Use negative prompt (only when changing category)')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level')
    parser.add_argument('--stop_on_exception', action='store_true', help='Stop on exception')
    parser.add_argument('--num_inference_steps_min', type=int, default=10, help='Min. Number of inference steps')
    parser.add_argument('--num_inference_steps_max', type=int, default=50, help='Max. Number of inference steps')
    parser.add_argument('--seed_num_inference_steps', type=int, default=26091993, help='Random seed for num_inference_steps (between min and max)')
    parser.add_argument('--num_images_per_prompt', type=int, default=3, help='Number of images per prompt')
    parser.add_argument('--do_nima', action='store_true', help='Do Aesthetic Assessment (default using only the bbox area)')
    parser.add_argument('--do_nima_on_full', action='store_true', help='Do Aesthetic Assessment using full 512 crop instead of only bbox area')
    parser.add_argument('--do_giqa', action='store_true', help='Do Aesthetic Assessment (default using only the bbox area)')
    parser.add_argument('--do_giqa_on_full', action='store_true', help='Do Aesthetic Assessment using full 512 crop instead of only bbox area')
    parser.add_argument('--do_itm', action='store_true', help='Do Image-Text Matching')
    parser.add_argument('--seed_gen', type=int, default=26091993, help='Random seed for generations')
    parser.add_argument('--fix_seed_gen', action='store_true', help='Always use the same seed for generations')

    parser.add_argument('--skip_if_exists', action='store_true', help='Skip generation if the combined generated files already exist')
    parser.add_argument('--skip_dir', type=str, help='Directory in which we should look for already-existing combined generated files')

    args = parser.parse_args()

    print("Arguments:")
    print(args)

    # Loadi inpainting model
    load_pipeline(args.inpainting_model_name, cache_dir=args.cache_dir)
    
    # Load COCO
    load_coco_all(args.coco_data_dir, args.coco_data_type)

    # Load NIMA
    if args.do_nima:
        load_nima(args.nima_model_path, args.vgg16_model_path)

    # Load GIQA
    if args.do_giqa:
        load_giqa(args.giqa_pca_path, args.giqa_pca_gmm_path)

    # For Image-Text Matching
    if args.do_itm:
        load_blip_itm(args.cache_dir)

    # Split categories
    categories = args.categories.split(",")
    categories_new = args.categories_new.split(",")
    if len(categories_new) == 0:
        categories_new = ["" for c in categories] # empty category_new (no text guiding)
    for category, category_new in zip(categories, categories_new):
        if not category:
            print("Empty category")
            continue
        print("Starting loop for %s -> %s" % (category, category_new))
        
        #directory_name = "%s_%s" % (datetime.datetime.now().strftime("%Y%m%d_%H%M%S"), category.replace(" ", "_"))
        if category == category_new:
            directory_name = category
        else:
            directory_name = "%s_%s" % (category, category_new)


        save_dir = "%s/%s" % (args.save_dir, directory_name) if args.save_dir else None
        print("Save dir: %s" % save_dir)
        save_orig_dir = "%s/%s" % (args.save_orig_dir, directory_name)  if args.save_orig_dir else None
        print("Save orig dir: %s" % save_orig_dir)
        bak_orig_dir = "%s/%s" % (args.bak_orig_dir, directory_name)  if args.bak_orig_dir else None
        print("Backup orig dir: %s" % bak_orig_dir)
        skip_dir = "%s/%s" % (args.skip_dir, directory_name) if args.skip_dir else None
        print("Skip dir: %s" % skip_dir)
        ps_mask_dir = "%s/%s" % (args.ps_mask_dir, directory_name) if args.ps_mask_dir else None
        print("PS Mask dir: %s" % ps_mask_dir)
    
        do_inpainting_loop(args.inpainting_model_name, category, category_new,
                           use_prompt_from_caption=args.use_prompt_from_caption, use_negative_prompt=args.use_negative_prompt,
                           max_images=args.max_images,
                           verbose=args.verbose, save_dir=save_dir, save_orig_dir=save_orig_dir, bak_orig_dir=bak_orig_dir,
                           use_bbox_mask=args.use_bbox_mask, use_segm_mask=args.use_segm_mask,
                           use_all_masks=args.use_all_masks,
                           use_ps_mask_for_splicing=args.use_ps_mask_for_splicing, ps_mask_dir=ps_mask_dir,
                           num_inference_steps_min=args.num_inference_steps_min, num_inference_steps_max=args.num_inference_steps_max,
                           num_images_per_prompt=args.num_images_per_prompt,
                           seed_num_inference_steps=args.seed_num_inference_steps, seed_gen=args.seed_gen, fix_seed_gen=args.fix_seed_gen,
                           do_nima=args.do_nima, do_itm=args.do_itm, do_nima_on_full=args.do_nima_on_full, do_giqa=args.do_giqa, do_giqa_on_full=args.do_giqa_on_full,
                           skip_if_exists=args.skip_if_exists, skip_dir=skip_dir,
                           stop_on_exception=args.stop_on_exception)

if __name__ == "__main__":
    main()
