
# rm -rf output/SMPLH_meshes/tsdf-rgbd/*
# rm -rf output/SMPLH_meshes/moved-meshes/*
# rm -rf output/SMPL_meshes/*
# rm -rf output/Close-output/*
# mv output/3DGS_meshes/* output/backup_meshes/

# echo "(*****************************************)"
# echo "1- Inferring 3DGS using HUMAN3DDIFFUSION... "
# cd Human3Diffusion
# python infer.py --checkpoints checkpoints
# python infer_mesh.py --checkpoints checkpoints --mesh_quality medium
# cd ../
# echo "Done!"
# echo "(*****************************************)"

# echo "(*****************************************)"
# echo "2- Fitting SMPLH using NICP... "
# cd NICP
# PYTHONPATH=. python ./src/lvd_templ/evaluation/evaluation_benchmark.py core.scaleback=True
# cd ../
# mv output/SMPLH_meshes/tsdf-rgbd/out_ss_cham_0.ply output/SMPLH_meshes/moved-meshes/out_ss.ply
# echo "Done!"
# echo "(*****************************************)"

# echo "(*****************************************)"
# echo "3- Fitting Converting SMPLH to SMPL... "
# cd smplx
# python -m transfer_model --exp-cfg config_files/smplh2smpl.yaml
# cd ../
# python scripts/plytoobj.py
# echo "Done!"
# echo "(*****************************************)"


echo "(*****************************************)"
echo "4- Preparing Scan for CloSeNet... "
cd CloSe
python prep_scan.py --garment_class Tshirt Hair Shoes Body Hat Shirt Vest Coat Dress Skirt Pants ShortPants Hoodies Swimwear Underwear Scarf Jumpsuits Jacket
echo "Done!"
echo "(*****************************************)"


echo "(*****************************************)"
echo "5- Running CloSeNet... "
python demo.py --render
cd ../
echo "Done!"
echo "(*****************************************)"

mv inputs/images/* inputs/sample-images/


echo "(*****************************************)"
echo "5- Creating GIFs... "
python /home/stud220/git/ImageTo3DSegmentedClothes/scripts/gifmaker.py
echo "Done!"
echo "(*****************************************)"
