import os
from PIL import Image
import sys
from pathlib import Path
import glob

def process_ffhq():
    # Set up the dataset directory
    home_dir = str(Path.home())
    dataset_dir = os.path.join(home_dir, "datasets", "ffhq")
    save_dir = os.path.join(home_dir, "datasets", "ffhq-128-png")
    
    print(f"Processing FFHQ dataset from: {dataset_dir}")
    print(f"Saving PNG images to: {save_dir}")
    
    try:
        # Create directory to store PNG images
        os.makedirs(save_dir, exist_ok=True)
        
        # Get all image files from the dataset
        image_files = glob.glob(os.path.join(dataset_dir, "**/*.png"), recursive=True)
        total_images = len(image_files)
        
        if total_images == 0:
            print("No PNG images found in the dataset directory!")
            print("Please make sure the FFHQ dataset is downloaded correctly.")
            sys.exit(1)
            
        print(f"Found {total_images} images to process...")
        
        # Convert and save images with progress tracking
        for idx, img_path in enumerate(image_files):
            if idx % 1000 == 0:
                print(f"Progress: {idx}/{total_images} images processed ({(idx/total_images)*100:.1f}%)")
            
            try:
                # Open and resize image to 128x128
                im = Image.open(img_path)
                im = im.resize((128, 128), Image.Resampling.LANCZOS)
                # Save with sequential numbering
                im.save(os.path.join(save_dir, f"{idx:05}.png"))
            except Exception as e:
                print(f"Error processing image {img_path}: {str(e)}")
                continue
        
        print("\nProcessing complete!")
        print(f"PNG images location: {save_dir}")
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    process_ffhq()