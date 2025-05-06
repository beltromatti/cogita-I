import argparse
import json
from src.inference import chat_inference
from src.utils import download_model
from src.app import launch_app

def main():
    """Main entry point for the Cogita AI 1.3b Project"""
    parser = argparse.ArgumentParser(description="Cogita AI 1.3b Machine Learning Coding Assistant")
    parser.add_argument("--download", action="store_true", help="Download the model")
    parser.add_argument("--inference", action="store_true", help="Simple inference mode")
    parser.add_argument("--app", action="store_true", help="Start the Cogita AI App")
    parser.add_argument("--config", type=str, default="config/config.json", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)

    if args.download:
        download_model("beltromatti/cogita-I")
    elif args.inference:
        while True:
            prompt = input("Hi! how can i help you? : ")
            chat_inference(prompt)
    elif args.app:
        launch_app()

if __name__ == "__main__":
    main()