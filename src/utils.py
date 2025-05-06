from transformers import AutoModelForSequenceClassification, AutoTokenizer

def download_model(model_name):

    # Download model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Save the model
    model.save_pretrained("./model")
    tokenizer.save_pretrained("./model")