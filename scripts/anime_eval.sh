pool="SNGAN_AnimeFaces_4"
eps=1
shift_steps=10
shift_leap=1
# =====================

declare -a EXPERIMENTS=("experiments/wip/SNGAN_AnimeFaces-LeNet-K16-D10-LearnGammas-eps0.25_0.35")

python sample_gan.py --num-samples 4 --pool "SNGAN_AnimeFaces_4" -g "SNGAN_AnimeFaces" \

wait

for exp in "${EXPERIMENTS[@]}"
do
  # Traverse latent space
  python traverse_latent_space.py -v --gif \
                                  --exp="${exp}" \
                                  --pool=${pool} \
                                  --eps=${eps} \
                                  --shift-steps=${shift_steps} \
                                  --shift-leap=${shift_leap}

done
