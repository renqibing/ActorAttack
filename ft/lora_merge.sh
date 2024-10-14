export CUDA_HOME=/mnt/petrelfs/share/test-cuda/cuda-12.1
BASE_LLAMA="/mnt/hwfile/trustai/lijun/models/llama3-8b-instruct/"
BASE_DIR="work_dirs/llama3_8b_instruct_qlora/"
BASE_PY="llama3_8b_instruct_qlora.py"
BASE_ITER="iter_$1"
srun -p AI4Good_L --gres=gpu:1 -J trans xtuner convert pth_to_hf "${BASE_DIR}${BASE_PY}" "${BASE_DIR}${BASE_ITER}.pth" "${BASE_DIR}${BASE_ITER}_hf"
srun -p AI4Good_L --gres=gpu:1 -J merge xtuner convert merge  $BASE_LLAMA "${BASE_DIR}${BASE_ITER}_hf" "${BASE_DIR}merged"