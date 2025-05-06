from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "model/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=False, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_auth_token=False, local_files_only=True).to(device)

def chat_inference(prompt):
    messages=[
        { 'role': 'user', 'content': prompt}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=2048, do_sample=True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id) #max_new_tokens=512
    print(tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True))

def code_generator_inference(prompt):
    input_text = "#"+str(prompt)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_length=128)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))