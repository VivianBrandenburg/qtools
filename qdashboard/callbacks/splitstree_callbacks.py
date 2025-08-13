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
    path_results =  meta_selected['path']
    
    # if no cropping folder exists yet, then create one 
    cropping_folder = path_results + '/tree_images_cropped/'
    if not os.path.exists(cropping_folder):
        os.makedirs(cropping_folder)
        
    # create all missing cropped images
    crop_images_in_folder(path_results + '/')
    
    # read the cropped version of the splitstree graph
    splitstree = cropping_folder + str(selected_epoch) + '.png'
    encoded_image = base64.b64encode(open(splitstree, 'rb').read())
    new_image_src = 'data:image/png;base64,{}'.format(encoded_image.decode())
    return new_image_src




