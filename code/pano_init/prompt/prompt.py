import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import os
import glob
import re
from PIL import Image
from .fix_videollama3_import import fix_videollama3_import 


class Lamma_Video:
    def __init__(self,device):
        self.device = device
        # Transformers>=4.56 moved VideoInput to video_utils; patch before remote imports run.
        fix_videollama3_import(verbose=False)
        self.model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path = "DAMO-NLP-SG/VideoLLaMA3-7B",
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",)
        # Model loading can refresh the cached remote files, so patch again before processor load.
        fix_videollama3_import(verbose=False)
        self.processor = AutoProcessor.from_pretrained("DAMO-NLP-SG/VideoLLaMA3-7B", trust_remote_code=True)
        

    def extract_prompt(self,img_path, debug=False):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": {"image_path": img_path}},
                    {"type": "text", "text":
                        "Generate a high-quality prompt starting with 'a' for Stable Diffusion. "
                        "Use only concise, non-repeating, comma-separated phrases. "
                        "Follow this structure: a [style+color+material keywords] [main subject], [details], (quality tags) . "
                        # "Avoid repetition. Example: a futuristic city, neon-lit buildings, bustling street, cinematic lighting, vibrant colors."
                    }
                ]
            }
        ]

   
        inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
   
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=200,              
                num_beams=4,                     
                temperature=1.2,                 
                repetition_penalty=1.2,          
                no_repeat_ngram_size=2,          
                early_stopping=True
            )
            
        response = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        response = self.clean_keywords(response)
        if debug:
            print(response)
        return response

    def clean_keywords(self, text):
        text = re.sub(r'[\(\[].*?[\)\]]', '', text)
        keywords = [k.strip() for k in text.split(',')]
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw and kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)

        return ', '.join(unique_keywords)

    def refine_prompt(self, raw_text):
        import re
        from collections import OrderedDict
 
        content = re.sub(r"\b(Stable|Diffusion)\b", "", raw_text, flags=re.IGNORECASE)  
        content = re.sub(r",{2,}", ",", content) 
        def advanced_sanitize(text):
            text = re.sub(r"[^a-zA-Z0-9(),:]+", ",", text)
            text = re.sub(r"(?<!,)\b(a)\b(?![\w])", "", text, flags=re.IGNORECASE)  
            tags = [t.strip() for t in text.split(",") if t.strip() and len(t) > 1]
            return list(OrderedDict.fromkeys(tags)) 
        
  
        base_tags = advanced_sanitize(content)
        
 
        quality_suffix = [
            "(best quality:1.2)",
            "4k",
            "8k",
            "highres",
            "masterpiece",
            "ultra-detailed",
            "(realistic:1.3)"
        ]
        
   
        final_tags = []
        has_a = False
        
        for tag in base_tags:
            if not has_a and re.match(r"^a\s+[\w]", tag, re.IGNORECASE):
                final_tags.append(tag)
                has_a = True
            elif not tag.startswith(("a ", "(best")):
                final_tags.append(tag)
  
        if not has_a:
            final_tags.insert(0, "a mesmerizing scene")
        
 
        return ", ".join(
            [tag for tag in final_tags if not re.fullmatch(r"a+", tag)] +  
            quality_suffix
        )[:512]  





def png2jpg(img_folder):
    all_files = glob.glob(os.path.join(img_folder, '**/*.*'), recursive=True)
    all_files = [f for f in all_files if f.lower().endswith('.png')]
    for png_file in all_files:
        jpg_file = os.path.splitext(png_file)[0] + ".jpg"
        Image.open(png_file).convert("RGB").save(jpg_file)
        if os.path.exists(jpg_file):
            os.remove(png_file)  # 删除原PNG
        print(f"转换完成: {png_file} -> {jpg_file}")

