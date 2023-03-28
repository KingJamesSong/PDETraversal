gan_type="VAE_MNIST"
dsprites=false
num_support_sets=16
num_support_timesteps=10
batch_size=256
max_iter=10000
tensorboard=true
# ================================

tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi

python train_vae_pretrained.py $tb \
                --gan-type=${gan_type} \
                --num-support-sets=${num_support_sets} \
                --num-support-timesteps=${num_support_timesteps} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100 \
                --dsprites=${dsprites}
