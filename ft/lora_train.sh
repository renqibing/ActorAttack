export CUDA_HOME=/mnt/petrelfs/share/test-cuda/cuda-12.1
srun -p AI4Good_0 --gres=gpu:4 -J train xtuner train llama3_8b_instruct_qlora.py --ft_dir $1 --launcher slurm