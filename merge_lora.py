from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import sys
# sys.path.append('/home/pretrain/hub-llama/') 


model_name_or_path = sys.argv[1]
adapter_name_or_path = sys.argv[2]
save_path = sys.argv[3]

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    # device_map={'':7}
)
print("load model success")
model = PeftModel.from_pretrained(model, adapter_name_or_path)
print("load adapter success")
model = model.merge_and_unload()
print("merge success")

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("save done.")

# python merge_lora.py 'TheBloke/Llama-2-13B-fp16' 'LLaMA-Efficient-Tuning/output/checkpoint-1000' 'models/'
