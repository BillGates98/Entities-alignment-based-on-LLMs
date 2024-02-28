# each suffix have 3 files : source, target, valid_same_as

# module load python/3.8.8
# python3.8 -m pip install modelscope
# python3.8 -m pip install sentencepiece
# python3.8 -m pip install transformers # git+https://github.com/huggingface/transformers
# python3.8 -m pip install accelerate

for dataset in 'doremus' # 'restaurant' 'person' 'SPIMBENCH_small-2019' 'SPIMBENCH_large-2016'
do
    for model in 'mistral' # 'gpt' 'qwen' 'bloom'
    do
        python3.8 ./main.py --input_path ./data/ --output_path ./outputs/ --suffix $dataset --dimension 200 --embedding r2v --llm_name $model --co_sim 0.7
    done
done

# git@github.com:BillGates98/Entities-alignment-based-on-LLMs.git