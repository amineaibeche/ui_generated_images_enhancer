import torch
import torch.nn as nn
from vision_transformer import VisionTransformer
import torch
from transformers import CLIPProcessor, CLIPModel
from diffusers import AutoPipelineForInpainting
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from django.conf import settings
import os

class ErrorMapPredictor(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super(ErrorMapPredictor, self).__init__(*args, **kwargs)

        # CLIP Text Feature Projection
        self.text_projection = nn.Linear(512, 768)

        # MOS Alignment Feature Projection
        self.mos_projection = nn.Sequential(
            nn.Linear(1, 256),  # Maps MOS Alignment to a 256D feature
            nn.ReLU(),
            nn.Linear(256, 768)  # Maps to match CLIP & vision feature dimensions
        )

        # Multi-modal Attention Fusion (Text + Image + MOS Alignment)
        self.fusion_layer = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        # Enhanced Feature Processing for Error Map
        self.conv_final = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)  # Output 1 channel error map
        )

        # Additional Residual Block for Feature Enhancement
        self.residual_block = nn.Sequential(
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(768, 768, kernel_size=3, padding=1),
        )

        # Regression Head for MOS Alignment
        self.regression_mos_align = nn.Sequential(
            nn.Linear(768, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 1),
        )

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        attn_weights = []
        patch_list = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)
            patch_list.append(x[:, 1:])

        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights, patch_list

    def forward(self, image, text_features, mos_align):
        x_cls, x_patch, attn_weights, patch_list = self.forward_features(image)

        # Convert text features to same dimension as vision features
        text_features = self.text_projection(text_features)

        # Convert MOS alignment into a feature representation
        mos_features = mos_align.unsqueeze(1)  # Shape: (B, 1)
        mos_features = self.mos_projection(mos_features)  # Shape: (B, 768)

        # Fusion: Vision + Text + MOS Alignment using Multi-modal Attention
        fusion_input = torch.cat([x_cls.unsqueeze(1), text_features.unsqueeze(1), mos_features.unsqueeze(1)], dim=1)
        fusion_output, _ = self.fusion_layer(fusion_input, fusion_input, fusion_input)

        # Extract enhanced features for error map and MOS alignment prediction
        enhanced_cls = fusion_output[:, 0]  # Shape: (B, 768)

        # Apply residual block for feature enhancement
        residual_features = self.residual_block(x_patch.permute(0, 2, 1).reshape(image.size(0), 768, int(x_patch.size(1) ** 0.5), int(x_patch.size(1) ** 0.5)))
        x_patch = x_patch + residual_features.flatten(2).permute(0, 2, 1)

        # Dynamically reshape `x_patch` for spatial processing
        batch_size, num_patches, embed_dim = x_patch.shape
        feature_size = int(num_patches ** 0.5)  # Compute dynamic feature map size
        x_patch = x_patch.permute(0, 2, 1).reshape(batch_size, embed_dim, feature_size, feature_size)

        # Generate Enhanced Error Map
        error_map = self.conv_final(x_patch)
        error_map = error_map + enhanced_cls.unsqueeze(-1).unsqueeze(-1)  # Expand cls token to match feature map

        # Predict MOS Alignment Score
        mos_align_pred = self.regression_mos_align(enhanced_cls)

        return error_map, mos_align_pred, attn_weights
    


# Image Preprocessing
def preprocess_image(image_path , device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0).to(device)

# Extract CLIP Text Features
def get_text_features(prompt, device , clip_processor , clip_model):
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    return text_features

# Generate **Inverted** Mask for Inpainting
# Generate Binary Mask for Inpainting
def generate_inpainting_mask(error_map_np, threshold=0.55):
    # Calculate the percentage of pixels exceeding the threshold
    total_pixels = error_map_np.size
    above_threshold_pixels = (error_map_np > threshold).sum()
    percentage_above_threshold = above_threshold_pixels / total_pixels

    # If less than 70% of the pixels exceed the threshold, use the entire error map as the mask
    if percentage_above_threshold < 0.65:
        mask = np.ones_like(error_map_np, dtype=np.uint8) * 255  # Full mask
    else:
        # Generate binary mask based on the threshold
        mask = (error_map_np > threshold).astype(np.uint8) * 255
        mask = 255 - mask  # Invert the mask

    # Resize the mask to match the image dimensions
    return cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)


def perform_inpainting(image, mask, text_prompt , guidance_scale , steps , trengh , device , pipe):
    generator = torch.Generator(device=device).manual_seed(0)
    inpainted_image = pipe(
        prompt=text_prompt,
        image=image,
        mask_image=mask,
        guidance_scale=guidance_scale,
        num_inference_steps=steps,
        strength=trengh,
        generator=generator
    ).images[0]
    return inpainted_image


def error_map_align_infer(image_path ,prompt ,  mos ,  mos_align , steps, strengh , guidance_scale ):
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model for text feature extraction
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Load the trained error map model
    model = ErrorMapPredictor().to(device)
    model.load_state_dict(torch.load(r"E:\Amine\PFE\IQA\yazid_idea\best_error_map_model_cnt_zelmati.pth", map_location=device))
    model.eval()

    # Load Stable Diffusion Inpainting Model
    pipe = AutoPipelineForInpainting.from_pretrained(
        "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
        torch_dtype=torch.float16,
        variant="fp16"
    ).to(device)

    image, image_tensor = preprocess_image(image_path , device)
    text_features = get_text_features(prompt , device , clip_processor , clip_model)
    mos_align = torch.tensor([mos_align], dtype=torch.float32, device=device)
    with torch.no_grad():
        error_map, mos_align_pred, _ = model(image_tensor, text_features, mos_align)
    
    error_map_np = error_map.squeeze(0).mean(dim=0).cpu().numpy()
    smoothed_error_map = gaussian_filter(error_map_np, sigma=2)   
    normalized_error_map = (smoothed_error_map - smoothed_error_map.min()) / (smoothed_error_map.max() - smoothed_error_map.min())
    binary_mask_resized = generate_inpainting_mask(normalized_error_map, 0.55)
    # Convert mask to PIL format for inpainting
    mask_pil = Image.fromarray(binary_mask_resized)

    # Perform inpainting
    inpainted_image = perform_inpainting(image, mask_pil, prompt,guidance_scale , steps , strengh , device , pipe)

    # Save the enhanced image with the same name
    enhanced_image_name = "enhanced_image.jpg"
    enhanced_image_path = os.path.join(settings.MEDIA_ROOT, 'enhanced', enhanced_image_name)
    os.makedirs(os.path.dirname(enhanced_image_path), exist_ok=True)  # Ensure the directory exists
    inpainted_image.save(enhanced_image_path)
    
    # Return the relative path with forward slashes
    relative_path = f"enhanced/{enhanced_image_name}".replace("\\", "/")
    del pipe 
    del clip_model
    del clip_processor
    del model
    return relative_path