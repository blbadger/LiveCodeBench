python -m lcb_runner.runner.main \
--model meta-llama/Meta-Llama-3.1-8B-Instruct \
--local_model_path "/home/bbadger/llama-3.1-8b-codeforcescots-qlora-b128/merged_model" \
--scenario codegeneration \
--cot_code_execution \
--evaluate \
--dtype float16 \
--max_tokens 8192 \
--tensor_parallel_size 4 \
--n 1 \
--release_version release_v1 \
--temperature 0.6 \
--top_p 0.95
#--use_cache

