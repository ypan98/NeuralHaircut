#!/bin/bash
colmap_data_folder=$1 # eg ../Datasets/usc_colmap
case=$2 # eg 00050

colmap_case_folder=$colmap_data_folder/$case
nh_data_folder=$colmap_case_folder/neural_haircut
nh_data_folder_sparse=$nh_data_folder/sparse_txt
pixie_folder=../PIXIE   # Path to PIXIE repo folder

start_time=$(date +%s)

# Create folder for neural haircut data
Check if the directory exists and delete it if it does
if [ -d "$nh_data_folder" ]; then
    echo "Directory $nh_data_folder already exists. Deleting..."
    rm -r $nh_data_folder
    echo "Directory deleted."
fi
Create folder again and populate with neural_haircut data
echo "Creating directory $nh_data_folder"
mkdir $nh_data_folder
mkdir $nh_data_folder_sparse

# Stage 1
echo "************************ Preprocessing for STAGE 1 ************************"
echo "Step 1: Converting raw COLMAP to txt"
colmap model_converter --input_path $colmap_data_folder/sparse/0  --output_path $nh_data_folder_sparse --output_type TXT
echo "Step 2: Parsing txt files to neural haircut format"
python preprocess_custom_data/colmap_parsing.py --path_to_scene  $nh_data_folder --save_path $nh_data_folder
echo "Step 3: Scaling scene into sphere"
python preprocess_custom_data/scale_scene_into_sphere.py --path_to_scene $nh_data_folder
echo "Step 4: Calculating masks"
python preprocess_custom_data/calc_masks.py --path_to_scene $nh_data_folder
echo "Step 5: Calculating orientation maps"
python preprocess_custom_data/calc_orientation_maps.py --img_path $nh_data_folder/image/ --orient_dir $nh_data_folder/orientation_maps --conf_dir $nh_data_folder/confidence_maps
echo "Step 6: Flame fitting"
echo "Step 6.1: Pixie Initialization"
python $pixie_folder/demos/pixie_initialization.py --inputpath $nh_data_folder/image/ --savefolder $nh_data_folder
echo "Step 6.2: Multiview Optimization"
python src/multiview_optimization/fit.py --data_path $nh_data_folder --save_path ./experiments/fit_person_1_bs_1/$case
python src/multiview_optimization/fit.py --data_path $nh_data_folder --batch_size 5 --save_path  ./experiments/fit_person_1_bs_5/$case --checkpoint_path ./experiments/fit_person_1_bs_1/$case/opt_params
python src/multiview_optimization/fit.py --data_path $nh_data_folder --batch_size 20 --train_shape True --save_path  ./experiments/fit_person_1_bs_20_train_rot_shape/$case  --checkpoint_path ./experiments/fit_person_1_bs_5/$case/opt_params
echo "Step 6.3: Copying fitted flame to $nh_data_folder/head_prior.obj"
fitted_flame=$(ls ./experiments/fit_person_1_bs_20_train_rot_shape/mesh/ | sort -n | tail -n 1)
cp ./experiments/fit_person_1_bs_20_train_rot_shape/$case/mesh/$fitted_flame $nh_data_folder/head_prior.obj
echo "Step 7: Cut eyes of FLAME head, needed for scalp regularizaton"
python preprocess_custom_data/cut_eyes.py --path_to_scene $nh_data_folder

echo "************************ STAGE 1: Geometry Reconstruction ************************"
python run_geometry_reconstruction.py --dataset_path $colmap_data_folder --case $case --conf ./configs/monocular/neural_strands.yaml

# Stage 2
echo "************************ Preprocessing for STAGE 2 ************************"
echo "Step 1: Copy the checkpoint for hair sdf and orientation field, obtained meshes to ./implicit-hair-data/data/$case"
python preprocess_custom_data/copy_checkpoints.py --case $case --path_to_scene $nh_data_folder/
echo "Step 2: Extract visible hair surface from sdf"
python preprocess_custom_data/extract_visible_surface.py --conf ./configs/monocular/neural_strands.yaml  --dataset_path $colmap_data_folder --case $case --img_size 2160 --n_views 2
echo "Step 3: Remesh hair_outer.ply to ~10k vertex for acceleration"
python preprocess_custom_data/remesh.py --path_to_head ./implicit-hair-data/data/$case/final_head.ply --path_to_hair ./implicit-hair-data/data/$case/hair_outer.ply
echo "Step 4: Extract scalp region for diffusion using the distance between hair sdf to scalp"
python preprocess_custom_data/cut_scalp.py --distance 0.07 --conf ./configs/monocular/neural_strands.yaml  --dataset_path $colmap_data_folder --case $case

echo "************************ STAGE 2: Strand Optimization ************************"
python run_strands_optimization.py --dataset_path $colmap_data_folder --case $case --conf ./configs/monocular/neural_strands.yaml  --hair_conf ./configs/hair_strands_textured.yaml

# End
end_time=$(date +%s)
execution_time=$((end_time - start_time))
echo "Total execution time: $execution_time seconds"