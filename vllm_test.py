from vllm import LLM, SamplingParams

prompts = [
    "Prove the pythagorean formula, a^2 + b^2 = c^2",
    "What is the capital of France?",
] 
sampling_params = SamplingParams(temperature=0.6)

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct", local_model="/experiments/llama-3.1-8b-codeforcescots/model")
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")