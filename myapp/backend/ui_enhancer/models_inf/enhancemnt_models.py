#the  prompt  enhancemnt  model 

import torch
import torch.nn as nn
import clip
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from PIL import Image
import torch.nn.functional as F
import pandas as  pd 
from tqdm.auto import tqdm
from diffusers import StableDiffusionImg2ImgPipeline
from transformers import CLIPProcessor, CLIPModel
from django.conf import settings
import os


class PromptEnhancementModel(nn.Module):
    def __init__(self, clip_model="ViT-B/32", llm_model="t5-base", max_length=50):
        super(PromptEnhancementModel, self).__init__()

        # Device setup
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load CLIP Model
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)

        # Load LLM (T5) for prompt generation
        self.tokenizer = T5Tokenizer.from_pretrained(llm_model)
        self.llm = T5ForConditionalGeneration.from_pretrained(llm_model).to(self.device)
        self.max_length = max_length

        # Projection layers to map CLIP features + MOS into LLM input space
        self.image_proj = nn.Linear(512, 768)  # CLIP image embedding -> LLM space
        self.text_proj = nn.Linear(512, 768)  # CLIP text embedding -> LLM space
        self.mos_proj = nn.Linear(3, 768)  # MOS quality, MOS alignment, and CLIP similarity -> LLM space

        # Fusion layer to combine features
        self.fusion = nn.Linear(768 * 3, 768)

    def forward(self, image, prompt, mos_quality, mos_align, input_ids, attention_mask, labels):
        """
        Improved model with CLIP alignment verification and adaptive enhancement strength.
        """
        # Normalize MOS scores from [0,5] to [0,1]
        mos_quality = mos_quality / 5.0
        mos_align = mos_align / 5.0

        # Ensure all inputs are on the same device
        image = image.to(self.device)
        mos_quality = mos_quality.to(self.device)
        mos_align = mos_align.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        labels = labels.to(self.device)

        # Extract CLIP features (image)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image).float()
            image_features = self.image_proj(image_features)  # Map to LLM space

        # Extract CLIP features (text)
        text_inputs = clip.tokenize(prompt).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs).float()
            text_features = self.text_proj(text_features)  # Map to LLM space

        # **Calculate Image-Text Similarity for Alignment Verification**
        similarity = F.cosine_similarity(image_features, text_features, dim=-1).unsqueeze(-1)  # Shape: (batch, 1)

        # Process MOS Scores & Similarity
        mos_features = torch.cat([mos_quality, mos_align, similarity], dim=-1)
        mos_features = self.mos_proj(mos_features)

        # Ensure batch size consistency
        batch_size = image_features.shape[0]
        text_features = text_features.expand(batch_size, -1)
        mos_features = mos_features.expand(batch_size, -1)

        # Feature Fusion
        fused_features = torch.cat([image_features, text_features, mos_features], dim=-1)
        fused_features = self.fusion(fused_features).unsqueeze(1)

        # Wrap fused features in BaseModelOutput for T5
        encoder_outputs = BaseModelOutput(last_hidden_state=fused_features)

        # Forward pass through T5
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return outputs.loss

    def generate(self, image, prompt, mos_quality, mos_align):
        """
        Generates an enhanced prompt using CLIP alignment and MOS quality.
        """
        self.eval()

        # Normalize MOS scores from [0,5] to [0,1]
        if isinstance(mos_quality, torch.Tensor):
            if mos_quality.numel() > 1:  # If batch, take the first element
                mos_quality = mos_quality[0].item()
            else:
                mos_quality = mos_quality.item()
        mos_quality = mos_quality / 5.0  # Scale to [0,1]

        if isinstance(mos_align, torch.Tensor):
            if mos_align.numel() > 1:
                mos_align = mos_align[0].item()
            else:
                mos_align = mos_align.item()
        mos_align = mos_align / 5.0  # Scale to [0,1]

        # Ensure `prompt` is a string (fix list issue)
        if isinstance(prompt, list):
            prompt = prompt[0]  # Take the first string in the list

        # Ensure image is a PIL Image before processing
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:  # If batch of images, take the first one
                image = image[0]  
            image = image.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
            image = Image.fromarray((image * 255).astype('uint8'))  # Convert to PIL Image

        # Preprocess image correctly
        image = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Extract CLIP features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image).float()
            text_inputs = clip.tokenize(prompt).to(self.device)  # Ensure `prompt` is a string
            text_features = self.clip_model.encode_text(text_inputs).float()

        # Calculate Image-Text Similarity (handle batch properly)
        similarity = F.cosine_similarity(image_features, text_features, dim=-1)  # Output shape: (batch_size,)
        similarity = similarity.mean().item()  # Convert to scalar

        # Adjust Enhancement Strength
        if mos_quality < 0.3 or mos_align < 0.3 or similarity < 0.6:
            enhancement_instruction = "Make it extremely detailed"
        elif 0.3 <= mos_quality < 0.6 or 0.3 <= mos_align < 0.6:
            enhancement_instruction = "Enhance clarity, add visual richness, but keep natural."
        else:
            enhancement_instruction = "Refine slightly, improving quality"

        # Generate the prompt
        input_prompt = f"Enhance: {prompt}. {enhancement_instruction}"

        input_ids = self.tokenizer(
            input_prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        ).input_ids.to(self.device)

        with torch.no_grad():
            output_ids = self.llm.generate(input_ids, max_length=self.max_length)

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)


def infer_prompt_ehancement(image_path , prompt , mos , mos_align , steps , guidance_scale , strengt):
    model = PromptEnhancementModel()
    model_path = r"E:\Amine\PFE\IQA\prompt_enhancement\best_prompt_model.pth"
    model.load_state_dict(torch.load(model_path, map_location="cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    image = Image.open(image_path).convert("RGB")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enhanced_prompt = model.generate(image, prompt, mos, mos_align)
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None
    ).to(device)
    with torch.no_grad():
        enhanced_image = pipe(
            prompt=enhanced_prompt,
            image=image,
            strength=strengt,
            guidance_scale=8.5,
            num_inference_steps=steps
        ).images[0]
    del model
    del pipe
    # Save the enhanced image in the MEDIA_ROOT/enhanced/ directory
    # Save the enhanced image in the MEDIA_ROOT/enhanced/ directory
    enhanced_image_name = "enhanced_image.jpg"
    enhanced_image_path = os.path.join(settings.MEDIA_ROOT, 'enhanced', enhanced_image_name)
    os.makedirs(os.path.dirname(enhanced_image_path), exist_ok=True)  # Ensure the directory exists
    enhanced_image.save(enhanced_image_path)
    
    # Return the relative path with forward slashes
    relative_path = f"enhanced/{enhanced_image_name}".replace("\\", "/")
    return relative_path