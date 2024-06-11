# %% [markdown]
# # Task 1: Training a Large Language Model
# - **Objective**: Train a Large Language Model using the provided dataset. The LLM should be capable of generating PlantUML code for a given scenario (which is an input to the LLM).
# - **Platform**: The training can be conducted on Google Colab.
# - **Deliverable**: A trained LLM that can successfully generate PlantUML code from scenario descriptions. Please upload the weights of the LLM on HuggingFace after training the LLM.

# %%
#!pip install datasets bitsandbytes accelerate peft # Comment it out when not using Colab
USER_ACCESS_TOKEN = "hf_..." #Deleted private token

# %%
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig
from datasets import load_dataset

# %% [markdown]
# First, we load the provided dataset in Huggingface [coai/plantuml_generation](https://huggingface.co/datasets/coai/plantuml_generation). The selected model to fine-tune is **Microsoft Phi 1.5**. Some preprocessing to the dataset is neccesary, making use of the tokenizer. As no evaluation split for the dataset is provided, this is left empty.

# %%
raw_dataset = load_dataset("coai/plantuml_generation", "default", split="train")#.select(range(16))

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", use_fast=True)

# Add special character to the tokenizer, this also tells the model when to stop
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(data):
    return tokenizer(data["text"], truncation=True)

# Map the full dataset with the tokenizer
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

eval_dataloader = None #No evaluation dataset provided

# %% [markdown]
# Define the [QLoRA (Dettmers et al. 2023)](https://arxiv.org/abs/2305.14314) byte configuration. This is done in order to enable large models to be trained in consumer hardware. In this case, a 4Bit quantization is used. This configuration is adapted for modern GPUs, in my case, an RTX 3060 6GB is used. [Source](https://kaitchup.substack.com/p/phi-2-a-small-model-easy-to-fine) 

# %%
compute_dtype = getattr(torch, "bfloat16")
bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=compute_dtype,
                                bnb_4bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", 
                                             trust_remote_code=True, 
                                             quantization_config=bnb_config, 
                                             device_map="auto",
                                             torch_dtype="auto")
model = prepare_model_for_kbit_training(model)
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj","k_proj","v_proj","fc2","fc1"] #Network layers that will be affected
)

model = get_peft_model(model, peft_config)

# %% [markdown]
# Finally, define the training hyperparameters and additional configuration to upload the model directly to Huggingface. Some of the hyperparameters have been obtained from the aforementioned source and [this one](https://medium.com/@amodwrites/a-definitive-guide-to-qlora-fine-tuning-falcon-7b-with-peft-78f500a1f337). We define the trainer and start training.

# %%
training_arguments = TrainingArguments(
        output_dir="finetuned_phi_15_plantuml_generation",
        push_to_hub=True,
        hub_token=USER_ACCESS_TOKEN,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=12,
        log_level="debug",
        save_steps=100,
        logging_steps=25, 
        learning_rate=1e-4,
        optim='paged_adamw_8bit',
        bf16=True, #change to fp16 if you are using an older GPU
        num_train_epochs=3,
        warmup_steps=100,
        lr_scheduler_type="linear",
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_arguments,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()

# %% [markdown]
# We can save again locally and in the hub the fine-tuned model after it finished. Additionally, the tokenizer is also uploaded. The model weights and tokenizer can be found under my [Huggingface page (becares)](https://huggingface.co/becares/finetuned_phi_15_plantuml_generation)

# %%
save_directory = "finetuned_phi_15_plantuml_generation"
model.save_pretrained(save_directory, push_to_hub=True, token=USER_ACCESS_TOKEN)
tokenizer.save_pretrained(save_directory, push_to_hub=True, token=USER_ACCESS_TOKEN) 

# %% [markdown]
# We may try the model directly giving it inputs. Make sure to run the imports and tokenizer cells in case you want to try the model without running the trainig.

# %%
config = PeftConfig.from_pretrained("becares/finetuned_phi_15_plantuml_generation")
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
model = PeftModel.from_pretrained(base_model, "becares/finetuned_phi_15_plantuml_generation")

inputs = tokenizer("Generate a plantuml diagram...", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


