gan_type="SNGAN_AnimeFaces"
num_support_sets=32
num_support_timesteps=20
reconstructor_type="LeNet"
batch_size=128
max_iter=120000
tensorboard=true
# ================================


tb=""
if $tensorboard ; then
  tb="--tensorboard"
fi

python train.py $tb \
                --gan-type=${gan_type} \
                --reconstructor-type=${reconstructor_type} \
                --num-support-sets=${num_support_sets} \
                --num-support-timesteps=${num_support_timesteps} \
                --batch-size=${batch_size} \
                --max-iter=${max_iter} \
                --log-freq=10 \
                --ckp-freq=100
