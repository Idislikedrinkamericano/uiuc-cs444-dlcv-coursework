import torch
import torchvision.transforms.functional as TF
from PIL import Image
from diffusion_helpers import add_variance
from diffusion_basics import one_step_denoise,  iterative_denoise_cfg

# Note: In all function signatures below, B refers to batch size in tensor shapes

# make_flip_illusion

def make_flip_illusion(image, i_start, prompt_embeds_1, prompt_embeds_2, timesteps, alphas_cumprod, stage_1, uncond_prompt_embeds, scale=7, display=True):
	with torch.no_grad():
		for i in range(i_start, len(timesteps) - 1):
			t = timesteps[i]
			prev_t = timesteps[i+1]
			
			# Calculate DDPM parameters
			alpha_cumprod = alphas_cumprod[t].to(image.device)
			alpha_cumprod_prev = alphas_cumprod[prev_t].to(image.device)
			alpha = alpha_cumprod / alpha_cumprod_prev
			beta = 1 - alpha
			
			# Get noise estimate for normal orientation
			model_output_cond = stage_1.unet(
				image, t, encoder_hidden_states=prompt_embeds_1, return_dict=False
			)[0]
			model_output_uncond = stage_1.unet(
				image, t, encoder_hidden_states=uncond_prompt_embeds, return_dict=False
			)[0]
			model_output_1 = model_output_uncond + scale * (model_output_cond - model_output_uncond)
			noise_est_1, pred_var_1 = torch.split(model_output_1, image.shape[1], dim=1)
			
			# Get noise estimate for flipped orientation
			image_flipped = torch.flip(image, dims=[2, 3])
			model_output_cond_2 = stage_1.unet(
				image_flipped, t, encoder_hidden_states=prompt_embeds_2, return_dict=False
			)[0]
			model_output_uncond_2 = stage_1.unet(
				image_flipped, t, encoder_hidden_states=uncond_prompt_embeds, return_dict=False
			)[0]
			model_output_2 = model_output_uncond_2 + scale * (model_output_cond_2 - model_output_uncond_2)
			noise_est_2, _ = torch.split(model_output_2, image.shape[1], dim=1)
			noise_est_2 = torch.flip(noise_est_2, dims=[2, 3])
			
			# Combine the two noise estimates
			final_noise = (noise_est_1 + noise_est_2) / 2
			
			# Apply DDPM reverse step - 正确的公式！
			x0 = (image - torch.sqrt(1 - alpha_cumprod) * final_noise) / torch.sqrt(alpha_cumprod)
			
			pred_prev_image = (
				torch.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod) * x0 +
				torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * image
			)
			
			# Add variance
			image = add_variance(pred_var_1, t, pred_prev_image, stage_1)
			
		clean = image.cpu().detach().numpy()
	return clean