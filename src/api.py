from fastapi import FastAPI
from pydantic import BaseModel 
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from plantuml import PlantUML
import tempfile

# Create the app instance
app = FastAPI()

# Load model and tokenizer directly from Huggingface
tokenizer = AutoTokenizer.from_pretrained("becares/finetuned_phi_15_plantuml_generation")

config = PeftConfig.from_pretrained("becares/finetuned_phi_15_plantuml_generation")
base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
model = PeftModel.from_pretrained(base_model, "becares/finetuned_phi_15_plantuml_generation")

# Load the PlantUML server
server = PlantUML(url='http://www.plantuml.com/plantuml/img/')

# Declare the input data to send it through HTTP
class Input(BaseModel):
    scenario: str

# Base root function
@app.get("/")
async def read_root():
    return {"message": "Becares Coding-Challenge \n finetuned_phi_15_plantuml_generation"}

# Post request for the API.
# The model process the received input (a scenario to generate) and outputs a PlantUML code.
# Using a temporary file to save the obtained input in a file, which will then be used by the
# PlantUML server to create the diagram. The diagram is saved in the root folder with the "diagram.png" name.
@app.post("/predict/")
async def predict(input: Input):
    encoded_inputs = tokenizer(input.scenario, return_tensors="pt")
    model_output = model.generate(**encoded_inputs, max_new_tokens=200)
    decoded_output = tokenizer.decode(model_output[0], skip_special_tokens=True)
    
    with tempfile.NamedTemporaryFile("w") as fp:
        fp.write(decoded_output)
        success = server.processes_file(fp.name, outfile="diagram.png")

    return {"success": success}