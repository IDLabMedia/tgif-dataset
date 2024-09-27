import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

def create_output_path(output_folder, img_path, suffix, extension):
    # Create the necessary subdirectories
    new_path = os.path.join(output_folder, f"{os.path.splitext(img_path)[0]}_{suffix}.{extension}")
    os.makedirs(os.path.dirname(new_path), exist_ok=True)
    return new_path

def post_process_images(input_csv, output_folder, operations_list, output_csv):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    df = pd.read_csv(input_csv)
    
    new_image_paths = []

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
        img_path = row['image']
        img = Image.open(img_path)
        
        for operation in operations_list:
            if 'jpeg_q' in operation:
                quality = int(operation.split('jpeg_q')[1])
                output_jpeg_path = create_output_path(output_folder, img_path, f"jpeg_q{quality}", "jpg")
                img.save(output_jpeg_path, 'JPEG', quality=quality)
                
                # Decode JPEG to PNG
                decoded_img = Image.open(output_jpeg_path)
                output_png_path = create_output_path(output_folder, img_path, f"jpeg_q{quality}", "png")
                decoded_img.save(output_png_path, 'PNG')
                new_image_paths.append(output_png_path)
                
            elif 'webp_q' in operation:
                quality = int(operation.split('webp_q')[1])
                output_webp_path = create_output_path(output_folder, img_path, f"webp_q{quality}", "webp")
                img.save(output_webp_path, 'WEBP', quality=quality)
                
                # Decode WEBP to PNG
                decoded_img = Image.open(output_webp_path)
                output_png_path = create_output_path(output_folder, img_path, f"webp_q{quality}", "png")
                decoded_img.save(output_png_path, 'PNG')
                new_image_paths.append(output_png_path)
    
    df['image'] = new_image_paths
    df.to_csv(output_csv, index=False)

    print(f"Images processed and saved in {output_folder}")
    print(f"CSV updated and saved as {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Post-process images from a CSV with specified operations.")
    parser.add_argument('--input_csv', type=str, required=True, help='The input CSV file containing image paths.')
    parser.add_argument('--output_folder', type=str, required=True, help='The output folder to save processed images.')
    parser.add_argument('--operations', type=str, required=True, help='Comma-separated list of operations (e.g., "jpeg_q80,webp_q50").')
    parser.add_argument('--output_csv', type=str, required=True, help='The output CSV file to save updated paths.')
    
    args = parser.parse_args()

    operations_list = args.operations.split(',')
    
    post_process_images(args.input_csv, args.output_folder, operations_list, args.output_csv)

if __name__ == "__main__":
    main()
