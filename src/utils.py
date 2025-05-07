from transformers import AutoModelForCausalLM, AutoTokenizer

def download_model(model_name, save_directory="./model"):
    # download the model and the tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_safetensors=True,
        trust_remote_code=False,
        torch_dtype="bfloat16"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model and the tokenizer in .safetensors format
    model.save_pretrained(
        save_directory, 
        safe_serialization=True, 
        max_shard_size="100GB"
    )
    tokenizer.save_pretrained("./model")