#!pip install datasets bitsandbytes accelerate flash_attn # Comment it out when not using Colab
USER_ACCESS_TOKEN = "" #Deleted private token

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from datasets import load_dataset

raw_dataset = load_dataset("coai/plantuml_generation", "default", split="train")#.select(range(16))

tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", add_eos_token=True, use_fast=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(data):
    return tokenizer(data["text"], truncation=True)

tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

eval_dataloader = None #No evaluation dataset provided

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
        target_modules= ["q_proj","k_proj","v_proj","fc2","fc1"]
)

model = get_peft_model(model, peft_config)

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
        bf16=True, #change to fp16 if are using an older GPU
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
trainer.push_to_hub()

save_directory = "finetuned_phi_15_plantuml_generation"
model.save_pretrained(save_directory, push_to_hub=True, token=USER_ACCESS_TOKEN)
tokenizer.save_pretrained(save_directory, push_to_hub=True, token=USER_ACCESS_TOKEN) 

inputs = tokenizer("Generate a plantuml diagram", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


