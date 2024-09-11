#!/bin/bash

colmap_data_folder=$1
nh_data_folder=$colmap_data_folder/neural_haircut
nh_data_folder_sparse=$nh_data_folder/sparse_txt

# echo "######################## Preprocessing "########################
# Check if the directory exists and delete it if it does
if [ -d "$nh_data_folder" ]; then
    echo "Directory $nh_data_folder already exists. Deleting..."
    rm -r $nh_data_folder
    echo "Directory deleted."
fi
# Create folder again and populate with neural_haircut data
echo "Creating directory $nh_data_folder"
mkdir $nh_data_folder
mkdir $nh_data_folder_sparse
echo "Step 1: Converting raw COLMAP to txt"
colmap model_converter --input_path $colmap_data_folder/sparse/0  --output_path $nh_data_folder_sparse --output_type TXT
echo "Step 2: Parsing txt files to neural haircut format"
python preprocess_custom_data/colmap_parsing.py --path_to_scene  $nh_data_folder --save_path $nh_data_folder
echo "Step 3: Scaling scene into sphere"
python preprocess_custom_data/scale_scene_into_sphere.py --path_to_scene $nh_data_folder
echo "Step 4: Calculating masks"
python preprocess_custom_data/calc_masks.py --path_to_scene $nh_data_folder