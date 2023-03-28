declare -a EXPERIMENTS=("experiments/wip/SNGAN_AnimeFaces-LeNet-K32-D10-LearnGammas-eps0.0_0.1")


for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python gen_pairs.py --exp="${exp}" \

done
