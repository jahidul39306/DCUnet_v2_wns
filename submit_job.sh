#!/bin/bash
#SBATCH --job-name=dcunet_dns_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=train_output_%j.log

#  FORCE STABLE AUDIO BACKENDS
# (Prevents the crash where it only looks for TorchCodec)
export TORCHAUDIO_USE_BACKEND_DISPATCHER=1
export TORCH_AUDIO_BACKEND="sox_io"

echo " [$(date)] DNS Challenge Training Started"
WORKDIR="/ceph/home/student.aau.dk/gr27bw/P8-AVS-WNS/mini-project-unet4"
cd $WORKDIR

if [ -d "./unet_env" ]; then
    echo "Activating environment from ./unet_env"
    source unet_env/bin/activate
fi

#  Run Training
python3 complex_train.py
