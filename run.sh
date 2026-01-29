bash_file_name=$(basename $0)

DEBUG=False

for dataset in "imagenet_r" "imagenet"
do
      for seed in 0
      do
            for severity in 5
            do
                  for arch in "resnet50"
                  do
                        for tta_method in "Tent"
                        do
                        python L-CS.py \
                              -acfg configs/adapter/${dataset}/${tta_method}.yaml \
                              -dcfg configs/dataset/${dataset}.yaml \
                              -mcfg configs/models/${arch}.yaml \
                              TEST.ROUNDS 1 \
                              SEED $seed \
                              TEST.BATCH_SIZE 64 \
                              CORRUPTION.SEVERITY "[${severity}]" \
                              NOTE "test" \
                              DEBUG $DEBUG
                        done
                  done
            done
      done
done
