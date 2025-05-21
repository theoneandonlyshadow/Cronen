# Import torch before setting Triton environment variables
import torch
import os

# ===============================
# PATCH TRITON TO FIX int32 ERROR AND DISABLE TRITON FOR T4 GPUS
# ===============================
import builtins
builtins.int32 = int  # monkey-patch for Triton 'int32' NameError

# ===============================
# COMPLETELY DISABLE TRITON FOR T4 COMPATIBILITY
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Ensure we're using the correct GPU
os.environ["FLASH_ATTENTION"] = "0"
os.environ["TRITON_DISABLE_LINE_INFO"] = "1"  
os.environ["TRITON_AUTOTUNE_BACKEND"] = "none"
os.environ["TRITON_CACHE_DIR"] = "/tmp/triton_cache"
os.environ["DISABLE_TRITON"] = "1"  # Completely disable Triton
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # Help with memory fragmentation

# ===============================
# IMPORT UNSLOTH AFTER ENVIRONMENT SETUP
# ===============================
from unsloth import FastLanguageModel

max_seq_length = 512  # Further reduced to 512 for T4 compatibility
dtype = torch.float16  # Explicitly use float16 instead of auto detection
load_in_4bit = True

# Print available GPU memory to help with debugging
print(f"Available GPU: {torch.cuda.get_device_name(0)}")
print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"Available GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB used, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

# Try a smaller model that will fit on T4 GPU (3B instead of 8B)
model_name = "unsloth/Meta-Llama-3-8B-Instruct-GGUF"  # Smaller model better suited for T4

# Set device mapping to allow CPU offloading
from transformers import BitsAndBytesConfig

# Configure quantization with CPU offloading
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


# Import AutoModelForCausalLM to use with device_map
from transformers import AutoModelForCausalLM, AutoTokenizer

# First try loading with standard AutoModelForCausalLM with device mapping
try:
    print("Attempting to load model with CPU offloading...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",  # let bitsandbytes & transformers offload to CPU
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("openchat/openchat-3.5-0106", trust_remote_code=True)
    print("Successfully loaded model with standard transformers library")
    
    # Convert to Unsloth format if possible
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model=model,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            dtype=dtype
        )
        print("Successfully converted to Unsloth format")
    except Exception as e:
        print(f"Could not convert to Unsloth format, continuing with standard model: {e}")
        
except Exception as e:
    print(f"Error loading with AutoModelForCausalLM: {e}")
    
    # Fallback to direct Unsloth loading with a very small model
    print("Falling back to direct Unsloth loading with a very small model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "microsoft/phi-3-mini-4k-instruct"
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True
    )
    print("Successfully loaded the small model with Unsloth")

# Update PEFT to target appropriate modules for the phi model family
# Phi models have different parameter names than DeepSeek
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj", 
                    "wq", "wk", "wv", "wo",  # Add these for Phi models
                    "w1", "w2", "w3"],  # Add these for Phi models
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None
)

from datasets import load_dataset
dataset = load_dataset("theoneandonlyshadow/Cronen", split="train")

# Let's check the structure of the dataset to understand it better
print("Dataset features:", dataset.features)
print("First example:", dataset[0])

from unsloth.chat_templates import get_chat_template
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

# Getting the right chat template based on the loaded model
if "phi" in str(model.__class__).lower() or "phi" in str(type(model)).lower():
    print("Using phi-specific chat template formatting")
    # Define a Phi-specific formatter
    def formatting_prompts_func(example):
        messages = example["messages"]
        formatted_text = ""
        
        for message in messages:
            if message["role"].lower() == "user":
                formatted_text += f"Human: {message['content']}\n\n"
            elif message["role"].lower() == "assistant":
                formatted_text += f"Assistant: {message['content']}\n\n"
        
        return {"text": formatted_text.strip()}
else:
    print("Using generic chat template formatting with tokenizer")
    # For other models, try to get a chat template from the tokenizer
    try:
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")
        
        def formatting_prompts_func(example):
            try:
                return {
                    "text": tokenizer.apply_chat_template(
                        example["messages"], 
                        tokenize=False, 
                        add_generation_prompt=False
                    )
                }
            except Exception as e:
                print(f"Error applying chat template: {e}")
                # Fallback to simple formatting
                messages = example["messages"]
                formatted_text = ""
                for message in messages:
                    formatted_text += f"{message['role'].capitalize()}: {message['content']}\n\n"
                return {"text": formatted_text.strip()}
    except Exception as e:
        print(f"Could not use Unsloth chat templates: {e}")
        # Simple fallback formatter
        def formatting_prompts_func(example):
            messages = example["messages"]
            formatted_text = ""
            for message in messages:
                formatted_text += f"{message['role'].capitalize()}: {message['content']}\n\n"
            return {"text": formatted_text.strip()}

# Map without batching - use num_proc=1 for safer processing with custom templates
print("Applying chat formatting...")
dataset = dataset.map(formatting_prompts_func, num_proc=1)

# Let's check the formatted output to make sure it looks right
print("\nExample of formatted text:")
print(dataset[0]["text"])
print("\n" + "-"*50 + "\n")

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=1,  
        gradient_accumulation_steps=8,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=True,  # Always use fp16 for T4
        bf16=False,  # T4 doesn't support bf16
        logging_steps=1,
        optim="paged_adamw_8bit",  # Use paged optimizer to save memory
        weight_decay=0.01,
        lr_scheduler_type="constant",  # Simpler scheduler for short training
        seed=3407,
        output_dir="outputs",
        report_to="none",
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        dataloader_num_workers=0,  # Reduce overhead
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,  # Keep all columns to avoid issues
    ),
)

from unsloth.chat_templates import train_on_responses_only

# Define response markers based on model type
if "phi" in str(model.__class__).lower() or "phi" in str(type(model)).lower():
    # For Phi models
    instruction_part = "Human: "
    response_part = "Assistant: "
    print(f"Using Phi-specific markers: '{instruction_part}' and '{response_part}'")
else:
    # For other models, fall back to generic or ChatML
    instruction_part = "<|im_start|>user\n"
    response_part = "<|im_start|>assistant\n"
    print(f"Using generic markers: '{instruction_part}' and '{response_part}'")

try:
    trainer = train_on_responses_only(
        trainer,
        instruction_part=instruction_part,
        response_part=response_part
    )
    print("Successfully applied train_on_responses_only")
except Exception as e:
    print(f"Error applying train_on_responses_only: {e}")
    print("Continuing with full text training instead of response-only")

# Train the model with error handling and workaround for Phi model issues
try:
    print("\nStarting training...")
    
    # Monkey patch to fix the Phi model forward method conflict
    import inspect
    
    # Check if this is a Phi model and apply workaround
    if hasattr(model, 'model') and hasattr(model.model, 'forward'):
        original_forward = model.model.forward
        
        def patched_forward(*args, **kwargs):
            # Remove conflicting parameter if present
            if 'logits_to_keep' in kwargs:
                kwargs.pop('logits_to_keep')
            return original_forward(*args, **kwargs)
        
        model.model.forward = patched_forward
        print("Applied Phi model forward method patch")
    
    trainer_stats = trainer.train()
    print("\nTraining completed successfully!")
    
except Exception as e:
    print(f"\nError during training: {e}")
    print("Let's try with alternative training setup...")
    
    # Alternative training approach without some problematic features
    try:
        # Create a new trainer with simpler configuration
        trainer_simple = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset.select(range(1000)),  # Use smaller subset for testing
            dataset_text_field="text",
            max_seq_length=256,  # Further reduce sequence length
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=2,
                max_steps=10,  # Reduce steps for testing
                learning_rate=1e-4,
                fp16=True,
                logging_steps=1,
                optim="adamw_8bit",
                output_dir="outputs_simple",
                report_to="none",
                save_steps=10,
            ),
        )
        
        print("Trying simplified training...")
        trainer_stats = trainer_simple.train()
        print("Simplified training completed!")
        
    except Exception as e2:
        print(f"Simplified training also failed: {e2}")
        print("Training failed, but we'll still try to save the model")

# For evaluation and saving, we need to ensure we use the correct templating
if "phi" in str(model.__class__).lower() or "phi" in str(type(model)).lower():
    # For Phi models - don't use custom chat templates
    tokenizer.pad_token = tokenizer.eos_token
    messages = [{"role": "user", "content": "I am sad because I failed my Maths test today"}]
    
    # Format messages manually
    prompt = f"Human: {messages[0]['content']}\n\nAssistant: "
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    print("\nEvaluation prompt:")
    print(prompt)
else:
    # For other models - try to apply the correct chat template
    try:
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="chatml")
    except:
        pass

    tokenizer.pad_token = tokenizer.eos_token
    messages = [{"role": "user", "content": "I am sad because I failed my Maths test today"}]
    
    # Try to use chat template
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True
        ).to("cuda")
    except Exception as e:
        print(f"Error applying chat template: {e}")
        # Fallback to manual formatting
        prompt = f"User: {messages[0]['content']}\n\nAssistant: "
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        print("\nFallback evaluation prompt:")
        print(prompt)

# Set model to inference mode
FastLanguageModel.for_inference(model)

# Safety check to ensure we're handling inputs correctly
try:
    # Handle different input formats
    if isinstance(inputs, dict):
        input_ids = inputs.get("input_ids", inputs)
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
    else:
        # inputs is a tensor
        input_ids = inputs
        attention_mask = torch.ones_like(inputs)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Generate text
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=64,
        use_cache=True,
        temperature=0.6,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Get only the generated part (after the prompt)
    input_length = input_ids.shape[1]
    generated_ids = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    print("\nGenerated output:")
    print(generated_text)

except Exception as e:
    print(f"Error during text generation: {e}")
    print("Trying alternative approach...")
    
    # Alternative approach for generating text
    try:
        # Simple generation without complex parameters
        with torch.no_grad():
            if isinstance(inputs, dict):
                input_tensor = inputs["input_ids"] if "input_ids" in inputs else inputs
            else:
                input_tensor = inputs
                
            output = model.generate(
                input_tensor,
                max_length=input_tensor.shape[1] + 64,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        result = tokenizer.decode(output[0], skip_special_tokens=True)
        print("\nGenerated output (alternative method):")
        print(result)
        
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")
        print("Text generation failed - this may be due to model compatibility issues")

# Save the model with error handling
my_model = "Vermilion"
try:
    print(f"\nSaving model to {my_model}...")
    model.save_pretrained(my_model)
    tokenizer.save_pretrained(my_model)
    print(f"Model successfully saved to {my_model}")
except Exception as e:
    print(f"Error saving model: {e}")
    
    # Try alternative saving method
    try:
        print("Trying alternative saving method...")
        # Save as a standard Hugging Face model
        model.save_pretrained(my_model, safe_serialization=False)
        tokenizer.save_pretrained(my_model)
        print(f"Model saved with alternative method to {my_model}")
    except Exception as e2:
        print(f"Alternative saving also failed: {e2}")

print("\nProcess complete!")
