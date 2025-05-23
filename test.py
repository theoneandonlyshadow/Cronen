import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

# Display available GPU info
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"Available GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB used, {torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

# Path to your saved model
model_path = "Vermilion"

def clean_response(text, original_prompt=""):
    """Enhanced response cleaning function"""
    # Remove the original prompt if it appears at the start
    if original_prompt and text.startswith(original_prompt):
        text = text[len(original_prompt):].strip()
    
    # Remove common template artifacts and HTML-like content
    template_patterns = [
        r'<\|[^|]*\|>',  # Remove <|...| > patterns
        r'<[^>]*>',      # Remove HTML-like tags
        r'\|[^|]*\|',    # Remove |...| patterns
        r'<br\s*/?>', r'</?\w+[^>]*>',  # HTML cleanup
        r'\[start[^\]]*\]', r'\[end[^\]]*\]',  # Remove [start...] [end...] patterns
        r'template\s*\d+', r'conversation\s*using',  # Remove template references
        r'additional\s*information\s*about',  # Remove metadata text
    ]
    
    for pattern in template_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Remove roleplay continuations and artifacts
    roleplay_patterns = [
        "Human:", "Assistant:", "Alien:", "H:", "Q:", "A:", "USER:", "ASSISTANT:",
        "\n\nHuman", "\n\nAssistant", "\n\nAlien", "###", "---"
    ]
    
    for pattern in roleplay_patterns:
        if pattern in text:
            text = text.split(pattern)[0]
    
    # Clean up whitespace and newlines
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with single space
    text = text.strip()
    
    # Remove empty parentheses, brackets, or other artifacts
    text = re.sub(r'\(\s*\)|\[\s*\]|\{\s*\}', '', text)
    text = re.sub(r'[|<>]+\s*$', '', text)  # Remove trailing artifacts
    
    return text

def extract_thinking_and_response(text):
    """Extract thinking process and final response"""
    thinking_text = ""
    final_response = text
    
    # Check for different thinking patterns
    thinking_patterns = [
        (r'<thinking>(.*?)</thinking>', lambda m: (m.group(1).strip(), text.replace(m.group(0), '').strip())),
        (r'Thinking:\s*(.*?)(?:\n(?:A:|Answer:|Response:))(.*)', lambda m: (m.group(1).strip(), m.group(2).strip())),
        (r'I need to (.*?)(?:\n|\.|$)(.*)', lambda m: (f"I need to {m.group(1).strip()}", m.group(2).strip()))
    ]
    
    for pattern, extractor in thinking_patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            thinking_text, final_response = extractor(match)
            break
    
    return thinking_text, final_response

try:
    print(f"\n=== Loading model from {model_path} ===")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Set pad token if not defined
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✓ Tokenizer loaded successfully")
    
    # Load model optimized for T4 GPU
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print("✓ Model loaded successfully!")
    
    print("\n--- Model Info ---")
    print(f"Model type: {type(model).__name__}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    print(f"Vocabulary size: {len(tokenizer)}")
    
    # Check what format the model expects by looking at tokenizer
    print(f"BOS token: {tokenizer.bos_token}")
    print(f"EOS token: {tokenizer.eos_token}")
    print(f"Chat template: {getattr(tokenizer, 'chat_template', 'None')}")
    
    # Interactive loop for testing
    print("\n=== Vermilion Model Test ===")
    print("Type your messages below. Type 'exit' to quit.")
    
    # Keep track of which format works best
    working_format_index = 0
    format_success_count = [0] * 5
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break
        
        # Improved prompt formats with stricter control
        prompt_formats = [
            # Direct conversation format
            f"Human: {user_input}\nAssistant:",
            
            # Simple Q&A format
            f"Question: {user_input}\nAnswer:",
            
            # Instruction format
            f"Respond to this message: {user_input}\nResponse:",
            
            # Chat format
            f"User: {user_input}\nBot:",
            
            # Plain format
            user_input
        ]
        
        response_found = False
        best_response = ""
        best_thinking = ""
        
        # Try the most successful format first, then others
        format_order = sorted(range(len(prompt_formats)), key=lambda i: format_success_count[i], reverse=True)
        
        for format_idx in format_order:
            if response_found:
                break
            
            formatted_input = prompt_formats[format_idx]
            print(f"\nTrying format {format_idx+1}...")
            
            # Tokenize input
            inputs = tokenizer(formatted_input, return_tensors="pt")
            input_length = inputs.input_ids.shape[1]
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate response with conservative settings
            with torch.no_grad():
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=120,   # Moderate length
                        temperature=0.4,      # Lower temperature for more focus
                        top_k=25,            # Focused sampling
                        top_p=0.75,          # Focused sampling
                        do_sample=True,
                        repetition_penalty=1.4,  # Higher to prevent repetition
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1,
                        early_stopping=True,
                        length_penalty=0.9,   # Slight penalty for length
                        num_beams=1,         # No beam search for speed
                        # Add stopping criteria
                        bad_words_ids=[[tokenizer.encode(word, add_special_tokens=False)[0]] for word in ["<|", "|>", "<br>", "</"] if tokenizer.encode(word, add_special_tokens=False)]
                    )
                    
                    # Decode and clean the response
                    raw_response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
                    
                    # Clean the response
                    cleaned_response = clean_response(raw_response, "")
                    
                    # Extract thinking and final response
                    thinking_text, final_response = extract_thinking_and_response(cleaned_response)
                    
                    # Final cleanup
                    final_response = clean_response(final_response)
                    
                    # Validate response quality
                    if (final_response and 
                        len(final_response.strip()) >= 3 and 
                        not re.match(r'^[^\w]*$', final_response) and  # Not just punctuation
                        len(final_response.split()) >= 1):  # At least one word
                        
                        print(f"✓ Format {format_idx+1} worked!")
                        
                        # Display thinking if present and meaningful
                        if thinking_text and len(thinking_text.strip()) > 5:
                            print(f"🤔 Thinking: {thinking_text}")
                        
                        print(f"Vermilion: {final_response}")
                        
                        # Update success tracking
                        format_success_count[format_idx] += 1
                        working_format_index = format_idx
                        response_found = True
                        best_response = final_response
                        best_thinking = thinking_text
                        break
                    else:
                        print(f"Format {format_idx+1} produced poor response: '{final_response}'")
                        
                except Exception as gen_error:
                    print(f"Format {format_idx+1} failed: {gen_error}")
                    continue
        
        if not response_found:
            print("❌ All formats failed. Trying emergency fallback...")
            
            # Emergency fallback with minimal processing
            try:
                inputs = tokenizer(user_input, return_tensors="pt")
                
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.2
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # Remove original input from response
                emergency_response = response.replace(user_input, "").strip()
                emergency_response = clean_response(emergency_response)
                
                if emergency_response:
                    print(f"Emergency response: {emergency_response}")
                else:
                    print("Even emergency fallback failed to produce meaningful output.")
                
            except Exception as emergency_error:
                print(f"Emergency fallback failed: {emergency_error}")

except Exception as main_error:
    print(f"Error loading or using model: {main_error}")
    
    print("\nTrying alternative loading method...")
    try:
        # Alternative loading approach
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("Loading model with CPU first...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Move to GPU if possible
        if torch.cuda.is_available():
            try:
                print("Moving model to GPU...")
                model.to("cuda:0")
            except Exception as gpu_error:
                print(f"GPU move failed, staying on CPU: {gpu_error}")
        
        device = next(model.parameters()).device
        print(f"Model is on device: {device}")
        
        # Simple test with cleaning
        test_prompt = "Hello, how are you?"
        inputs = tokenizer(test_prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print("Testing basic generation...")
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )
        
        raw_test_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        test_response = clean_response(raw_test_response.replace(test_prompt, ""))
        print(f"Test result: {test_response}")
        
        if test_response and len(test_response.strip()) > 3:
            print("✓ Basic generation working with cleaning!")
            
            # Start simplified interactive mode
            print("\nStarting simplified interactive mode...")
            while True:
                user_input = input("\nYou: ")
                if user_input.lower() in ["exit", "quit", "q"]:
                    break
                
                # Use simple approach that worked in test
                inputs = tokenizer(user_input, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                try:
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.5,
                        top_k=30,
                        top_p=0.8,
                        do_sample=True,
                        repetition_penalty=1.4,
                        pad_token_id=tokenizer.eos_token_id,
                        num_return_sequences=1
                    )
                    
                    raw_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    clean_final_response = clean_response(raw_response.replace(user_input, ""))
                    
                    if clean_final_response:
                        print(f"Vermilion: {clean_final_response}")
                    else:
                        print("Vermilion: [No meaningful response generated]")
                        
                except Exception as simple_error:
                    print(f"Generation error: {simple_error}")
        else:
            print("❌ Model not generating properly even with alternative method")
        
    except Exception as alt_error:
        print(f"Alternative loading also failed: {alt_error}")
        print("\nDebug information:")
        if os.path.exists(model_path):
            files = os.listdir(model_path)
            print(f"Files in model directory: {files}")
            
            # Check for config files
            config_files = [f for f in files if 'config' in f.lower()]
            print(f"Config files found: {config_files}")
        else:
            print("Model directory not found")

print("\n=== Session Complete ===")
print("Format success rates:")
format_names = ["Human/Assistant", "Question/Answer", "Instruction", "User/Bot", "Plain"]
for i, (name, count) in enumerate(zip(format_names, format_success_count)):
    print(f"  {name}: {count} successes")
