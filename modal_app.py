import modal 
import os 
import zipfile
from pathlib import Path 
import torch 
from huggingface_hub import hf_hub_download

# Define the container image
app = modal.App("gemma-inference")
image = (
    modal.Image.debian_slim().uv_pip_install(
        "transformers",
        "torch",
        "accelerate",
        "peft",
        "bitsandbytes",
        "fastapi[standard]",
        "huggingface_hub"
    )
)

@app.cls(
    gpu="T4",
    image=image,
    max_containers=1,
    scaledown_window=180,
    secrets=[modal.Secret.from_name("hugging_face")]
)
class ModalPipeline:

    @modal.enter()
    def load_model(self) -> None:
        from transformers import Gemma2ForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, PeftModel

        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )   

        access_token = os.environ["HF_MODEL_TOKEN"]

        # Load the model's tokenizer and the pre-trained model
        self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it", token=access_token)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = Gemma2ForCausalLM.from_pretrained("google/gemma-2-9b-it", 
                                               quantization_config=config, 
                                               device_map="auto", 
                                               token=access_token
                                              )

        # Load in the weight path 
        zip_file = hf_hub_download(
            repo_id=os.environ["MODEL_REPO"],
            filename=os.environ["MODEL_FILE_NAME"],
            token=os.environ["HF_TOKEN"]
        )

        adapter_root = Path("/tmp/adapter")
        adapter_root.mkdir(parents=True, exist_ok=True)

        # Extract the zip path 
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
           zip_ref.extractall(adapter_root)

        config_files = list(adapter_root.rglob("adapter_config.json"))

        if not config_files:
           raise LookupError("Cannot find config file")

        adapter_pth = config_files[0].parent

        self.model = PeftModel.from_pretrained(
            self.base_model,
            str(adapter_pth),
            is_trainable=False,
            local_files_only=False,
        )

        self.model.eval()

    @modal.method()
    def generate_response(self, prompt: str) -> str:
        message = [
            {
                "role": "user",
                "content": prompt
            }
        ]


        model_input = self.tokenizer.apply_chat_template(
           message,
           tokenize=True,
           add_generation_prompt=True,
           return_tensors="pt",
           return_dict=True
        )

        input_device = self.model.get_input_embeddings().weight.device

        model_input = {
           key: value.to(input_device)
           for key, value in model_input.items()
        }

        with torch.inference_mode():
            output = self.model.generate(
                **model_input,
                max_new_tokens=318,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                num_beams=4,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3
            )

        input_length = model_input["input_ids"].shape[-1]
           
        generated_tokens = output[0][input_length:]
               
        return self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

@app.function(image=image)
@modal.fastapi_endpoint(method='POST', requires_proxy_auth=True)
def app_generate(data: dict) -> dict:
    prompt = data.get("prompt")

    if not isinstance(prompt, str):
        raise ValueError("Prompt is not a string")

    pipeline = ModalPipeline()
    response = pipeline.generate_response.remote(prompt)
    return {"response": response} 
