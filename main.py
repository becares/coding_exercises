from fastapi import FastAPI
from pydantic import BaseModel 
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from plantuml import PlantUML
import torch
import tempfile

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("becares/finetuned_phi_15_plantuml_generation")

config = PeftConfig.from_pretrained("becares/finetuned_phi_15_plantuml_generation")
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
model = PeftModel.from_pretrained(base_model, "becares/finetuned_phi_15_plantuml_generation")

server = PlantUML(url='http://www.plantuml.com/plantuml/img/')

class Input(BaseModel):
    scenario: str

@app.get("/")
async def read_root():
    return {"message": "Becares Coding-Challenge \n finetuned_phi_15_plantuml_generation"}

@app.post("/predict/")
async def predict(input: Input):
    encoded_inputs = tokenizer(input.scenario, return_tensors="pt")
    model_output = model.generate(**encoded_inputs, max_new_tokens=200)
    decoded_output = tokenizer.decode(model_output[0], skip_special_tokens=True)
    
    decoded_output = input.scenario
    
    with tempfile.NamedTemporaryFile("w") as fp:
        fp.write(decoded_output)
        success = server.processes_file(fp.name, outfile="diagram.png")

    return {"success": success}