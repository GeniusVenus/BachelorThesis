import numpy as np

# Function to convert mask to RGB
def mask_to_color_image(mask, color_map):
    color_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_index, color in color_map.items():
        color_image[mask == class_index] = color
    return color_image