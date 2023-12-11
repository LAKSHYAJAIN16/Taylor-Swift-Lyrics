# Get all of the Data
import os
import torch
import torch.nn as nn
import bitsandbytes as bnb
import transformers
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model 

subfolders = [f.path for f in os.scandir(r"C:\Users\USER\Projects\tay-llm\Taylor-Swift-Lyrics\data\Albums") if f.is_dir() ]
data = []
for fold in subfolders:
    data.append([])
    for path in os.listdir(fold):
        if os.path.isfile(os.path.join(fold, path)):
            # We got the file boiz!
            file = open(os.path.join(fold, path), 'r', encoding="utf8")
            lines = file.readlines()
            lines.pop(0)
            lines.pop(len(lines) - 1)
            string_1 = "Write a song in Taylor Swift's style called " + path.removesuffix(".txt").replace("TaylorsVersion","").replace("_"," ") + "->:!-> \n "
            string = string_1 + " ".join(lines)
            data.append({"prediction":string,"input":string_1})           

del data[0]
data =  data[:10]
print(data[0])

model = AutoModelForCausalLM.from_pretrained(
    "bigscience/bloom-7b1", 
    load_in_8bit=False, 
    device_map='auto',
)

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
for param in model.parameters():
  param.requires_grad = False  # freeze the model - train adapters later
  if param.ndim == 1:
    # cast the small parameters (e.g. layernorm) to fp32 for stability
    param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

class CastOutputToFloat(nn.Sequential):
  def forward(self, x): return super().forward(x).to(torch.float32)
model.lm_head = CastOutputToFloat(model.lm_head)

config = LoraConfig(
    r=16, #attention heads
    lora_alpha=32, #alpha scaling
    # target_modules=["q_proj", "v_proj"], #if you know the 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM" # set this for CLM or Seq2Seq
)

model = get_peft_model(model, config)

for m in range(0, len(data)):
    try:
        data[m]["prediction"] = tokenizer(data[m]["prediction"])
    except:
        print(m)

trainer = transformers.Trainer(
    model=model, 
    train_dataset=data,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=2,
        warmup_steps=100, 
        max_steps=200, 
        learning_rate=2e-4, 
        fp16=False,
        logging_steps=1, 
        output_dir='outputs'
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()