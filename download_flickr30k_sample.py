import os
import json
import random
from datasets import load_dataset
from tqdm import tqdm

def setup_directories():
    """Create necessary directories for saving images and metadata."""
    os.makedirs('flickr30k_sample/images', exist_ok=True)
    os.makedirs('flickr30k_sample/metadata', exist_ok=True)

def save_image(image_data, filename):
    """Save image data to file."""
    try:
        image_data.save(filename)
        return True
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
        return False

def main():
    setup_directories()

    print("Loading Flickr30k dataset...")
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    
    total_images = len(dataset)
    print(f"Total images available: {total_images}")
    
    sample_size = 500
    random_indices = random.sample(range(total_images), sample_size)
    
    print(f"\nDownloading {sample_size} random images...")
    for idx in tqdm(random_indices):
        item = dataset[idx]
        
        image_filename = f"flickr30k_sample/images/image_{idx}.jpg"
        metadata_filename = f"flickr30k_sample/metadata/image_{idx}.json"
        
        if save_image(item['image'], image_filename):
            metadata = {
                'image_id': item['img_id'],
                'filename': item['filename'],
                'captions': item['caption'],
                'split': item['split']
            }
            
            with open(metadata_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("\nDownload completed!")
    print("Images saved in: flickr30k_sample/images/")
    print("Metadata saved in: flickr30k_sample/metadata/")

if __name__ == '__main__':
    main() 