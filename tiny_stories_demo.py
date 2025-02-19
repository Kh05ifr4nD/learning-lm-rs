from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("raincandy-u/TinyStories-656K")
model = AutoModelForCausalLM.from_pretrained("raincandy-u/TinyStories-656K")


# 2. 设备配置
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32


# 4. 生成函数
def generate_story(prompt: str, max_length=800) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# 示例调用
if __name__ == "__main__":
    prompt = "Once upon a time"
    print(generate_story(prompt))
