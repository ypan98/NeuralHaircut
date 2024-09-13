#!/bin/bash
colmap_data_folder=$1 # eg ../Datasets/usc_colmap
case=$2 # eg 00050
colmap_case_folder=$colmap_data_folder/$case
nh_data_folder=$colmap_data_folder/neural_haircut
nh_data_folder_sparse=$nh_data_folder/sparse_txt

pixie_folder=../PIXIE   # Path to PIXIE repo folder

# # echo "######################## Preprocessing "########################
# Check if the directory exists and delete it if it does
# if [ -d "$nh_data_folder" ]; then
#     echo "Directory $nh_data_folder already exists. Deleting..."
#     rm -r $nh_data_folder
#     echo "Directory deleted."
# fi
# Create folder again and populate with neural_haircut data
# echo "Creating directory $nh_data_folder"
# mkdir $nh_data_folder
# mkdir $nh_data_folder_sparse

# echo "************************ Preprocessing for STAGE 1 ************************"
# echo "Step 1: Converting raw COLMAP to txt"
# colmap model_converter --input_path $colmap_data_folder/sparse/0  --output_path $nh_data_folder_sparse --output_type TXT
# echo "Step 2: Parsing txt files to neural haircut format"
# python preprocess_custom_data/colmap_parsing.py --path_to_scene  $nh_data_folder --save_path $nh_data_folder
# echo "Step 3: Scaling scene into sphere"
# python preprocess_custom_data/scale_scene_into_sphere.py --path_to_scene $nh_data_folder
# echo "Step 4: Calculating masks"
# python preprocess_custom_data/calc_masks.py --path_to_scene $nh_data_folder
# echo "Step 5: Calculating orientation maps"
# python preprocess_custom_data/calc_orientation_maps.py --img_path $nh_data_folder/image/ --orient_dir $nh_data_folder/orientation_maps --conf_dir $nh_data_folder/confidence_maps
# echo "Step 6: Flame fitting"
# echo "Step 6.1: Pixie Initialization"
# python $pixie_folder/demos/pixie_initialization.py --inputpath $nh_data_folder/image/ --savefolder $nh_data_folder
# echo "Step 6.2: Multiview Optimization"
# python src/multiview_optimization/fit.py --data_path $nh_data_folder --save_path ./experiments/fit_person_1_bs_1
# python src/multiview_optimization/fit.py --data_path $nh_data_folder --batch_size 5 --save_path  ./experiments/fit_person_1_bs_5 --checkpoint_path ./experiments/fit_person_1_bs_1/opt_params
# python src/multiview_optimization/fit.py --data_path $nh_data_folder --batch_size 20 --train_shape True --save_path  ./experiments/fit_person_1_bs_20_train_rot_shape  --checkpoint_path ./experiments/fit_person_1_bs_5/opt_params
# echo "Step 6.3: Copying fitted flame to head_prior.obj"
# fitted_flame=$(ls ./experiments/fit_person_1_bs_20_train_rot_shape/mesh/ | sort -n | tail -n 1)
# cp ./experiments/fit_person_1_bs_20_train_rot_shape/mesh/$fitted_flame $nh_data_folder/head_prior.obj
# echo "Step 7: Cut eyes of FLAME head, needed for scalp regularizaton"
# python preprocess_custom_data/cut_eyes.py --path_to_scene $nh_data_folder

echo "************************ STAGE 1: Geometry Reconstruction ************************"
CUDA_LAUNCH_BLOCKING=1 python run_geometry_reconstruction.py --dataset_path $colmap_data_folder --case $case --conf ./configs/monocular/neural_strands.yaml

# echo "************************ Preprocessing for STAGE 2 ************************"
# echo "Step 1: Copy the checkpoint for hair sdf and orientation field, obtained meshes to the scene folder for convenience"
# python ./preprocess_custom_data/copy_checkpoints.py --case CASE --exp_name first_stage_reconctruction_CASE --conf_path ./configs/SCENE_TYPE/neural_strands*.yaml
