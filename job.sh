# each suffix have 3 files : source, target, valid_same_as
#
for dataset in 'doremus' 'restaurant' 'person' 'SPIMBENCH_small-2019' 'SPIMBENCH_large-2016'
do
    for model in 'llama2' 'qwen' 'gpt'
    do
        python3.8 ./main.py --input_path ./data/ --output_path ./outputs/ --suffix $dataset --dimension 200 --embedding r2v --llm_name $model --co_sim 0.7
    done
done