from vllm import LLM, SamplingParams

prompts = [
    "You will be given a competitive programming problem. Please reason step by step about the solution, then provide a complete implementation in C++17.\n\nYour solution must read input from standard input (cin), write output to standard output (cout).\nDo not include any debug prints or additional output.\n\nPut your final solution within a single code block:\n```cpp\n<your code here>\n```\n\n# Problem\n\nYou are given an array $$$a$$$ of $$$n$$$ integers, where $$$n$$$ is odd.\n\nIn one operation, you will remove two adjacent elements from the array $$$a$$$, and then concatenate the remaining parts of the array. <think> Okay, "
] 
sampling_params = SamplingParams(temperature=0.6, max_tokens=1024)

llm = LLM(model="/home/bbadger/experiments/llama-3.1-8b-codeforcescots-qlora-b64/merged_model", 
	tensor_parallel_size=4,
	enable_chunked_prefill=False,
	max_model_len=16384,
	dtype='float16'
)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
