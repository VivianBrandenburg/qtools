#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from dash import  Input, Output, callback
import base64
import os 
from PIL import Image


# cropping images before plotting splitstree graphs. 
def crop_image(image_path):
    im = Image.open(image_path)
    right, lower = im.size
    left, upper = 0,0
    h1, h2 = 150, 250 
    
    path_prefix = '/'.join(image_path.split('/')[:-1]) + '/'
    newfile = path_prefix + 'cropped/' + image_path.split('/')[-1]
    
    im.crop((left+h1, upper+h2, right-h1, lower-h2)).save(newfile, quality=100)




def crop_images_in_folder(path): 
    original_images = [x for x in os.listdir(path) if x.endswith('.png')]
    relocated_original_images = [x for x in os.listdir(path+'/tree_images') if x.endswith('.png')]
    cropped_images = [x for x in os.listdir(path+'/tree_images_cropped') if x.endswith('.png')]
    missing_images = [x for x in original_images+relocated_original_images if x not in cropped_images ]
    for x in missing_images: 
        crop_image(path + x)





# show splitstrees
@callback(
    Output('splitstree-figure', 'src'),
    Input('meta_selected', 'data'),
    Input("epoch-selected", "value"))
def update_tree(meta_selected, selected_epoch):
    path_results = meta_selected['path']

    # Prefer 'tree_images_cropped', then 'cropped'
    cropping_folder = None
    for folder in [os.path.join(path_results, 'tree_images_cropped'), os.path.join(path_results, 'cropped')]:
        if os.path.isdir(folder):
            cropping_folder = folder
            break

    # If no cropped folder, but original images exist, auto-crop to 'tree_images_cropped'
    if cropping_folder is None:
        originals = [x for x in os.listdir(path_results) if x.endswith('.png')]
        if originals:
            cropping_folder = os.path.join(path_results, 'tree_images_cropped')
            if not os.path.exists(cropping_folder):
                os.makedirs(cropping_folder)
            for x in originals:
                cropped_path = os.path.join(cropping_folder, x)
                if not os.path.isfile(cropped_path):
                    crop_image(os.path.join(path_results, x))
        else:
            return 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="40"><text x="10" y="25" font-size="14" fill="gray">No cropped or original splitstree images found.</text></svg>'

    splitstree = os.path.join(cropping_folder, f'{selected_epoch}.png')
    if not os.path.isfile(splitstree):
        return 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="40"><text x="10" y="25" font-size="14" fill="gray">No splitstree image for this epoch.</text></svg>'

    with open(splitstree, 'rb') as f:
        encoded_image = base64.b64encode(f.read())
    new_image_src = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return new_image_src




