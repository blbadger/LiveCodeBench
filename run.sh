python -m lcb_runner.runner.main \
--model "meta-llama/Meta-Llama-3.1-8B" \
--local_model_path "/home/bbadger/Desktop/deepseek-llama-3.1-8b" \
--scenario codegeneration \
--cot_code_execution \
--evaluate \
--dtype float16 \
--max_tokens 4096 \
--tensor_parallel_size 4 \
--n 1 \
--release_version release_v5 \
--temperature 0.58 \
--top_p 0.95
#--use_cache

