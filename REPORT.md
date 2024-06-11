# AI Engineer / Prompt Engineer
This repository is for the Coding-Challenge in the application process of AI Engineer / Prompt Engineer HS Ansbach.
The aim of this project is to solve a problem in the area of Large Language Models (LLMs), combining both Backend and Frontend tasks. For further information, please contact me [here](mailto:abecares001@gmail.com).

## Installation and Environment Setup
1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Set up the environment**:

    Ensure you have Python and **Conda** installed. Create the environment, activate it and install required packages.
    ```bash
    conda create -n coding-exercises python=3.10.12
    conda activate coding-exercises
    pip install -r requirements.txt
    ```
    Alternatively, you can create the environment using **venv**, but please note you will need to change the environment loading accordingly in the run script.

3. **Run the API and Frontend loading script**:
   
   Run the activation bash script. If you are in Windows, remember to create an equivalent version for Powershell. Note that you may need to change permissions to allow script execution.
   ```bash
   chmod +x run.sh
   ./run.sh
   ```
   This will pop automatically the GUI webpage, as well as loading the API and LLM in the background. The script also automatically activates the Conda environment, so you can run it without activating it beforehand.

# Report

## Task 1: Training a Large Language Model
#### **Objective:** Train a Large Language Model using the provided dataset. The LLM should be capable of generating PlantUML code for a given scenario (which is an input to the LLM).

This first step involves training and fine-tuning a model with the provided dataset, in order to create a reliable model capable of interpreting the given scenarios and generate PLantUML code that will be further processed by the Backend API. The notebook and equivalent Python scripts can be found in ```src/train_llm.ipynb``` and ```src/train_llm.py```, respectively.

The model selected to be fine-tuned is the[**microsoft/phi-1_5**](https://huggingface.co/microsoft/phi-1_5) model, which is a light, non-restricted and open-source model suitable for research. Additionally, it is well suited to write Python code, which is a good start for fine-tuning PlantUML code generation. Other models where also tested, but due to hardware limitations, these were discarded. List of other tested models:
 - [mistralai/Codestral-22B-v0.1](https://huggingface.co/mistralai/Codestral-22B-v0.1)
 - [TinyLlama/1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
 - [llama-2-7b-hf-small-shards](https://huggingface.co/abhishek/llama-2-7b-hf-small-shards)
 - [microsoft/Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct)

For the training, a RTX 3060 Mobile/Max-Q (6GB) was used. The 3 epoch training fully using the provided dataset finished in approximately 2 hours. Due to time constrains and, again, hardware limitations, further training was not performed.

Additional remarks include the implementation of the ```transformers``` trainer pipeline for easy hyperparameter and Huggingface model upload, as well as the [QLoRA](https://arxiv.org/abs/2305.14314) finetuning approach to allow for training in customer hardware. Using the latest, a stable and "fast" training was achieved in under 3GB of GPU memory usage.

Model weights, tokenizer and metadata can be found in my personal Huggingface space: [becares](https://huggingface.co/becares/finetuned_phi_15_plantuml_generation)

## Task 2: Backend and Frontend Development
#### **Backend Objective**: Develop a backend service that generates PlantUML code from a given scenario and converts it into an image.

This second task involves using FastAPI to create a Backend API that will connect a Frontend GUI with the fine-tuned model. The code can be found in ```src/api.py```.

For this task, we need to create a FastAPI app instance, load the model + tokenizer and the PlantUML diagram generation server. Additionally, we create a data container that inherits the ```pydantic``` BaseModel to store the data that will be send using HTTP afterwards.

Two asynchronous functions are then declared. The first one is the Base root function for FastAPI, and the second one, the only request that will be needed for the API interaction. Further information can found in the code documentation.

The API can be launched simply using the terminal with ```fastapi run src/api.py```. This will create a ```uvicorn``` instance that can be accessed in http://0.0.0.0:8000 (or localhost if using the dev launch).

#### **Frontend Objective**: Develop a frontend interface where users can input scenarios and view the generated PlantUML diagrams.

The final task involves the creation of a used-friendly GUI to interact with the Backend and LLM. In this case, instead of ReactJS/NodeJS, the Python ```streamlit``` package was used:
- I don't have much experience with ReactJS/NodeJS, apart from some small projects done during my Bachelor studies (about 3-4 years ago). Due to the time constraints to deliver this project, I couldn't invest much time in re-learning the syntax and create an acceptable GUI afterwards.
- Streamlit offers an easy integration with existing Python packages, as well as providing with a clean, open-source interface that mimics ChatBot apps. Therefore, a familiar and visually clean GUI can be created in few code lines.

As aforementioned, the interface offers a ChatBot-like view where an assistant prompts some instructions to use the app. After typing the desired scenario, the interface will communicate with the API to generate PlantUML code using the model, and then return the generated diagram using only one POST request. No further functionality was added to the GUI, and the assistant responses are not dynamic. More insight on the functionality can be found in the code documentation.

The frontend can be easily launched using ```streamlit run src/ui.py```. This will launch a localhost webpage that will open automatically.

## Useful links, Sources and References
[1] Huggingface personal space [becares](https://huggingface.co/becares)

[2] [QLoRA: Efficient Finetuning of Quantized LLMs (De)](https://arxiv.org/abs/2305.14314)

[3] [Phi-2: A Small Model Easy to Fine-tune on Your GPU](https://kaitchup.substack.com/p/phi-2-a-small-model-easy-to-fine)

[4] [Textbooks Are All You Need II: phi-1.5 technical report](https://arxiv.org/abs/2309.05463)

[5] [Huggingface](https://huggingface.co/), [FastAPI](https://fastapi.tiangolo.com/) and [Streamlit](https://streamlit.io/) Forums, Guides and Documentation...