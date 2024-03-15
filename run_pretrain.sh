#!/bin/bash
run=mae-lm

export update_freq=1
export batch_size=32
export tokens_per_sample=512
export max_tokens=$(($batch_size*$tokens_per_sample))
export lr=2e-4
export warmup_steps=10000
export bsm=$(($batch_size/4)) # this is to ensure the same iter-per-epoch for different batch sizes
export other_opts='--seed 100 --rel-pos 1 --gen-decoder-layers 4 --no-return-mask --max-rel-pos 128 --rel-pos-bins 64' 
export NCCL_SOCKET_IFNAME=ens32
save_dir=$1
data_name=$2

arch="ae"
task="masked_lm"
criterion="masked_lm"


if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  ddp_options=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
	ddp_options=""
  else
    ddp_options="--distributed-world-size 64 --distributed-port 12232"
  fi
fi
echo "ddp_options: ${ddp_options}"


mkdir -p $save_dir
touch $save_dir/train.log

python train.py $data_name --num-workers 8 --ddp-backend=c10d \
       --task $task --criterion $criterion \
       --arch $arch --sample-break-mode complete_doc --tokens-per-sample $tokens_per_sample \
       --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-6 --clip-norm 2.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update 2000000 \
       --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
       --max-tokens $max_tokens --update-freq $update_freq  --seed 1 \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 --tensorboard-logdir ${save_dir}/tsb \
       --max-update 2000000 --log-format simple --log-interval 100 --required-batch-size-multiple $bsm \
       --save-interval-updates 10000 --keep-interval-updates 30 --no-epoch-checkpoints --disable-validation --skip-invalid-size-inputs-valid-test \
       $ddp_options $other_opts --save-dir $save_dir | tee -a $save_dir/train.log