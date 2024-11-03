import pandas as pd

df = pd.read_csv('Users\Ayush Mishra\Downloads\archive\qa_Cell_Phones_and_Accessories.json')

# Display the first few rows
df.head()


from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import torch

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Prepare data in the format for model training
# This will depend on the structure of your conversations
conversations = [f"{q} {tokenizer.eos_token} {a} {tokenizer.eos_token}" for q, a in zip(df['Question'], df['Answer'])]

# Tokenize and create inputs for the model
inputs = tokenizer(conversations, return_tensors='pt', padding=True, truncation=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=2,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs'
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=inputs['input_ids']
)

# Fine-tune the model
trainer.train()


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    chat_history_ids = model.generate(input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5000)
