from transformers import AutoModelForSequenceClassification, AutoTokenizer

def download_model(model_name):
    # Scarica il modello e il tokenizer, forzando l'uso di safetensors
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, use_safetensors=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Salva il modello e il tokenizer in formato safetensors
    model.save_pretrained("./model", safe_serialization=True)
    tokenizer.save_pretrained("./model")