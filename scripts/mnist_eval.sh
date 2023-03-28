# === Configuration ===
pool="VAE_MNIST_10"
eps=1
shift_steps=10
shift_leap=1
# =====================

declare -a EXPERIMENTS=("experiments/wip/VAE_MNIST-LeNet-K16-D10-LearnGammas-eps0.15_0.25")

python sample_gan.py --num-samples 10 --pool "VAE_MNIST_10" -g "VAE_MNIST" \

wait

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python traverse_latent_space_vae.py -v --gif \
                                  --exp="${exp}" \
                                  --pool=${pool} \
                                  --eps=${eps} \
                                  --shift-steps=${shift_steps} \
                                  --shift-leap=${shift_leap}
done
