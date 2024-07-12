
backbones=("maxvit")
hidden_dims=(256)

additional_suffix="frameds"
# datasets=("volleyball" "kovo_video")
# data_paths=("data/volleyball/volleyball_mp4" "data/volleyball/2324/mp4")

datasets=("volleyball")
data_paths=("data/volleyball/volleyball_mp4")


# model_names=("tdetr2" "tdetr")
model_names=("maxvit2")
epochs=50
batch_size=2
dataset_mode="frame"
optim="adamw"
accumulate_grad_batches=16
imgz=512

# define image size of 224 by 224

# Iterate over each combination of parameters
for backbone in "${backbones[@]}"; do
  for hidden_dim in "${hidden_dims[@]}"; do
    for i in "${!datasets[@]}"; do
      dataset="${datasets[$i]}"
      data_path="${data_paths[$i]}"
      for model_name in "${model_names[@]}"; do
        log_suffix="${additional_suffix}_${backbone}_hidden_${hidden_dim}"
        echo "Running with model_name: $model_name, backbone: $backbone, hidden_dim: $hidden_dim, dataset: $dataset, data_path: $data_path, epochs: $epochs, batch_size: $batch_size, epoch_size: $epoch_size, log_suffix: $log_suffix"
        python main.py --model_name $model_name --epochs $epochs --use_temporal_encodings --batch_size $batch_size --dataset $dataset --data_path $data_path --accumulate_grad_batches $accumulate_grad_batches --backbone $backbone --optimizer $optim --hidden_dim $hidden_dim --log_subfix $log_suffix --dataset_mode $dataset_mode --num_workers 8  --imgsz 512 512
      done
    done
  done
done
