import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import math
from pathlib import Path
import csv
import torch
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
def create_output_path(output_folder, img_path, suffix, extension):
    # Create the necessary subdirectories
    new_path = os.path.join(output_folder, f"{os.path.splitext(img_path)[0]}_{suffix}.{extension}")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    return new_path


def downsample_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Downsample the image by the given scale factor.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    new_w = int(image.shape[1] * scale_factor)
    new_h = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

def upsample_image(image: np.ndarray, scale_factor: int, method: str = "cubic") -> np.ndarray:
    """
    Upsample the image by an integer scale factor (e.g., 2 or 4).

    Parameters:
        image (np.ndarray): Input image (H x W x C or H x W).
        scale_factor (int): Upscaling factor (e.g., 2, 4).
        method (str): Interpolation method: "cubic", "linear", "nearest", or "lanczos".
                      Default is "cubic".

    Returns:
        np.ndarray: Upscaled image.
    """
    if scale_factor <= 0:
        raise ValueError("scale_factor must be positive")
    if not isinstance(scale_factor, int):
        raise ValueError("scale_factor should be an integer (e.g., 2, 4)")

    interp_map = {
        "cubic": cv2.INTER_CUBIC,
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "lanczos": cv2.INTER_LANCZOS4,  # high quality but slower
    }

    if method not in interp_map:
        raise ValueError(f"Invalid method '{method}'. Choose from {list(interp_map.keys())}")

    new_w = image.shape[1] * scale_factor
    new_h = image.shape[0] * scale_factor

    return cv2.resize(image, (new_w, new_h), interpolation=interp_map[method])

def calculate_lpips(lpips_metric_calculator, img_gt, img_pred):
    img_pred = np.array(img_pred).astype(np.float32)/255.0
    img_gt = np.array(img_gt).astype(np.float32)/255.0
    assert img_pred.shape == img_gt.shape, f"Image shapes should be the same but are {img_pred.shape} and {img_gt.shape}."
        
    img_pred_tensor=torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(device)
    img_gt_tensor=torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(device)
        
    score =  lpips_metric_calculator(img_pred_tensor*2-1,img_gt_tensor*2-1)
    score = score.cpu().item()
    
    return score

def compute_psnr(img1, img2, max_val=255.0):
    img1 = np.array(img1).astype(np.float32)
    img2 = np.array(img2).astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(max_val) - 10 * np.log10(mse)
    
def calculate_psnr(img_pred, img_gt):
    img_pred = np.array(img_pred).astype(np.float32)/255.
    img_gt = np.array(img_gt).astype(np.float32)/255.
    assert img_pred.shape == img_gt.shape, f"Image shapes should be the same but are {img_pred.shape} and {img_gt.shape}."
    
    mse = np.mean((img_pred - img_gt) ** 2)
    
    if mse < 1.0e-10:
        return 1000
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def post_process_images(input_csv, output_folder, operations_list, output_csv,
                        do_calculate_psnr=False, do_calculate_lpips=False, skip_if_exists=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    df = pd.read_csv(input_csv)
    
    new_image_paths = []
    for operation in operations_list:
        new_image_paths.append([]) # list of lists
    psnrs = np.zeros((len(operations_list), df.shape[0]), dtype=np.float32)
    lpipses = np.zeros((len(operations_list), df.shape[0]), dtype=np.float32)
    
    lpips_metric_calculator = None
    if do_calculate_lpips:
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)

    realesrgan_args = {}
    upsamplers = {}
    for operation in operations_list:
        if 'realesrgan_x' in operation:
            scale_factor = int(operation.split('realesrgan_x')[1])
            assert scale_factor == 2 or scale_factor == 4, "Only realesrgan_x2 and realesrgan_x4 supported"
            import run_realesrgan
            realesrgan_args[operation] = run_realesrgan.RealESRGANArgs(model_name="RealESRGAN_x%dplus" % scale_factor, outscale=scale_factor)
            # No support for face enhancer
            upsamplers[operation] = run_realesrgan.load_upsampler(realesrgan_args[operation])

    Path(output_csv).parents[0].mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header_row = ["image"]
        for o_i, operation in enumerate(operations_list):
            header_row.append(f'path_{operation}')
            if do_calculate_psnr:
                header_row.append(f'psnr_{operation}')
            if do_calculate_lpips:
                header_row.append(f'lpips_{operation}')
        writer.writerow(header_row)
        
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
            img_path = row['image']
            #print(index, img_path)
            
            img = Image.open(img_path)
            img_opencv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            row_to_write = [img_path]
            for o_i, operation in enumerate(operations_list):
                #print(o_i, operation)
                new_image_path = ''
                if 'jpeg_q' in operation:
                    quality = int(operation.split('jpeg_q')[1])
                    output_png_path = create_output_path(output_folder, img_path, f"jpeg_q{quality}", "png")
                    if skip_if_exists and os.path.exists(output_png_path):
                        decoded_img = Image.open(output_png_path)
                    else:
                        output_jpeg_path = f"{img_path}_jpeg_q{quality}.jpg"
                        img.save(output_jpeg_path, 'JPEG', quality=quality)
                        
                        # Decode JPEG to PNG
                        decoded_img = Image.open(output_jpeg_path)
                        #output_png_path = create_output_path(output_folder, img_path, f"jpeg_q{quality}", "png")
                        decoded_img.save(output_png_path, 'PNG')
                    new_image_path = output_png_path
                    new_image = decoded_img
                    
                elif 'webp_q' in operation:
                    quality = int(operation.split('webp_q')[1])
                    output_png_path = create_output_path(output_folder, img_path, f"webp_q{quality}", "png")
                    if skip_if_exists and os.path.exists(output_png_path):
                        decoded_img = Image.open(output_png_path)
                    else:
                        output_webp_path = create_output_path(output_folder, img_path, f"webp_q{quality}", "webp")
                        img.save(output_webp_path, 'WEBP', quality=quality)
                        
                        # Decode WEBP to PNG
                        decoded_img = Image.open(output_webp_path)
                        output_png_path = create_output_path(output_folder, img_path, f"webp_q{quality}", "png")
                        decoded_img.save(output_png_path, 'PNG')
                    new_image_path = output_png_path
                    new_image = decoded_img
    
                elif 'realesrgan_x' in operation:
                    # https://github.com/xinntao/Real-ESRGAN
                    scale_factor = int(operation.split('realesrgan_x')[1])
                    output_path_up = create_output_path(output_folder, img_path, f"realersgan_x{scale_factor}", "png")
                    output_path_up_down = "%s_down.png" % output_path_up
                    if skip_if_exists and os.path.exists(output_path_up_down):
                        img_upsampled_downsampled = Image.open(output_path_up_down)
                        img_upsampled_downsampled = cv2.cvtColor(np.array(img_upsampled_downsampled), cv2.COLOR_RGB2BGR)
                    else:
                        # Upsample
                        #output_path_up = create_output_path(output_folder, img_path, f"realersgan_x{scale_factor}", "png")
                        img_upsampled = run_realesrgan.run_realesrgan(realesrgan_args[operation], upsamplers[operation], img_path, output_path=output_path_up)
        
                        # Downsample
                        #output_path_up_down = "%s_down.png" % output_path_up
                        img_upsampled_downsampled = downsample_image(img_upsampled, scale_factor=1.0/scale_factor)
                        #save_image(output_path_up_down, img_upsampled_downsampled)
                        cv2.imwrite(output_path_up_down, img_upsampled_downsampled)
                    new_image_path = output_path_up_down
                    new_image = img_upsampled_downsampled
    
                elif operation.startswith("up_"):
                    # Expected format: up_<method>_x<factor>
                    # Example: up_cubic_x2
                    try:
                        parts = operation.split("_")  # e.g. ["up", "cubic", "x2"]
                        method = parts[1]
                        scale_factor = int(parts[2][1:])  # remove "x" prefix, e.g. "x2" -> 2
                    except (IndexError, ValueError):
                        raise ValueError(f"Invalid operation format: {operation}. Expected 'up_<method>_x<factor>'")
    
                    output_path_up = create_output_path(output_folder, img_path, f"up_{method}_x{scale_factor}", "png")
                    output_path_up_down = "%s_down.png" % output_path_up
                    if skip_if_exists and os.path.exists(output_path_up_down):
                        img_upsampled_downsampled = Image.open(output_path_up_down)
                        img_upsampled_downsampled = cv2.cvtColor(np.array(img_upsampled_downsampled), cv2.COLOR_RGB2BGR)
                    else:
                        # Upsample
                        #output_path_up = create_output_path(output_folder, img_path, f"up_{method}_x{scale_factor}", "png")
                        img_upsampled = upsample_image(img_opencv, scale_factor=scale_factor, method=method)
                        cv2.imwrite(output_path_up, img_upsampled)
        
                        # Downsample
                        #output_path_up_down = "%s_down.png" % output_path_up
                        img_upsampled_downsampled = downsample_image(img_upsampled, scale_factor=1.0/scale_factor)
                        cv2.imwrite(output_path_up_down, img_upsampled_downsampled)
                    new_image_path = output_path_up_down
                    new_image = img_upsampled_downsampled      
                
                new_image_paths[o_i].append(new_image_path)          

                row_to_write.append(new_image_path)
                if do_calculate_psnr:
                    psnr = calculate_psnr(img, new_image)
                    psnrs[o_i, index] = psnr
                    row_to_write.append(psnr)
                if do_calculate_lpips:
                    lpips = calculate_lpips(lpips_metric_calculator, img, new_image)
                    lpipses[o_i, index] = lpips
                    row_to_write.append(lpips)
            writer.writerow(row_to_write)
                
    #print(new_image_paths)
    #df['image'] = new_image_paths
    #df = df = pd.DataFrame()
    #for o_i, operation in enumerate(operations_list):
        #df[f'path_{operation}'] = new_image_paths[o_i]
        #if do_calculate_psnr:
        #    df[f'psnr_{operation}'] = psnrs[o_i]
        #if do_calculate_lpips:
        #    df[f'lpips_{operation}'] = lpipses[o_i]
    #Path(output_csv).parents[0].mkdir(parents=True, exist_ok=True)
    #df.to_csv(output_csv, index=False)

    print(f"Images processed and saved in {output_folder}")
    print(f"CSV updated and saved as {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Post-process images from a CSV with specified operations.")
    parser.add_argument('--input_csv', type=str, required=True, help='The input CSV file containing image paths.')
    parser.add_argument('--output_folder', type=str, required=True, help='The output folder to save processed images.')
    parser.add_argument('--operations', type=str, required=True, help='Comma-separated list of operations (e.g., "jpeg_q80,webp_q50").')
    parser.add_argument('--output_csv', type=str, required=True, help='The output CSV file to save updated paths.')
    parser.add_argument('--skip_if_exists', action='store_true', help='Skip if exists (still calculate PSNR/LPIPS)')
    parser.add_argument('--do_calculate_psnr', action='store_true', help='Calculate PSNR')
    parser.add_argument('--do_calculate_lpips', action='store_true', help='Calculate LPIPS')
    
    args = parser.parse_args()

    operations_list = args.operations.split(',')
    
    post_process_images(args.input_csv, args.output_folder, operations_list, args.output_csv,
                       args.do_calculate_psnr, args.do_calculate_lpips, args.skip_if_exists)

if __name__ == "__main__":
    main()
