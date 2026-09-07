import torch
import torchvision.transforms.functional as TF
from PIL import Image
from diffusion_helpers import add_variance

# Note: In all function signatures below, B refers to batch size in tensor shapes

# forward

def forward(im, t, alphas_cumprod):
	"""
	Forward process: add noise to clean image
	Args:
		im: clean image tensor of shape (B, 3, 64, 64)
		t: timestep (integer scalar)
		alphas_cumprod: cumulative product of alphas of shape (1000,)
	Returns:
		noisy image of shape (B, 3, 64, 64)
	"""
	# TODO: Implement this function
	alpha_bar = alphas_cumprod[t].to(im.device)
	noise = torch.randn_like(im)
	im_noisy = torch.sqrt(alpha_bar) * im + torch.sqrt(1 - alpha_bar) * noise
	return im_noisy # noisy image tensor


def classical_denoise(im_noisy):
	"""
	Classical denoising: apply Gaussian blur to the noisy image
	Args:
		im_noisy: noisy image of shape (B, 3, 64, 64) or (3, 64, 64)
	Returns:
		denoised: denoised image of shape (B, 3, 64, 64) or (3, 64, 64)
	"""
	# TODO: Implement this function
	denoised = TF.gaussian_blur(im_noisy, kernel_size=5, sigma=1.5)
	return denoised # denoised image tensor


def one_step_denoise(im_noisy, noise_est, alpha_cumprod):
	"""
	One step denoising: remove noise from the noisy image to obtain an estimate of the original image
	Args:
		im_noisy: noisy image of shape (B, 3, 64, 64)
		noise_est: noise estimate tensor from the UNet (stage_1 output) of shape (B, 3, 64, 64)
		alpha_cumprod: alpha_cumprod scalar tensor or float
	Returns:
		clean_one_step: estimated original image of shape (B, 3, 64, 64)
	"""
	# TODO: Implement this function
	alpha_bar = alpha_cumprod.to(im_noisy.device)
	clean_one_step = (im_noisy - torch.sqrt(1 - alpha_bar) * noise_est) / torch.sqrt(alpha_bar)
	return clean_one_step



def iterative_denoise(image, i_start, prompt_embeds, timesteps, alphas_cumprod, stage_1):
	"""
	Iterative denoising: remove noise from the noisy image to obtain an estimate of the original image
	Args:
		image: noisy image of shape (B, 3, 64, 64)
		i_start: starting timestep index
		prompt_embeds: prompt embeddings of shape (B, 77, 4096)
		timesteps: list of timesteps
		alphas_cumprod: cumulative alphas of shape (1000,)
		stage_1: DeepFloyd stage_1 model
	Returns:
		clean: estimated original image of shape (B, 3, 64, 64)
		intermediate_images: list of intermediate images
	"""
	image = image.half().cuda()
	intermediate_images = []
	with torch.no_grad():
		for i in range(i_start, len(timesteps) - 1):
			# Get timesteps
			t = timesteps[i]
			prev_t = timesteps[i+1]

			# TODO: Get alphas, betas
			# ===== your code here! =====
			alpha_cumprod = alphas_cumprod[t].to(image.device)
			alpha_cumprod_prev = alphas_cumprod[prev_t].to(image.device)
			alpha = alpha_cumprod / alpha_cumprod_prev
			beta = 1 - alpha
			# ==== end of code ====

			# Store intermediate images every 5th step
			if (i - i_start) % 5 == 0:
				intermediate_images.append({
					'step': i - i_start + 1,
					'timestep': t,
					'image': image.cpu().detach().numpy()
				})

			# Get noise estimate
			model_output = stage_1.unet(
					image,
					t,
					encoder_hidden_states=prompt_embeds,
					return_dict=False
			)[0]

			# Split estimate into noise and variance estimate
			noise_est, predicted_variance = torch.split(model_output, image.shape[1], dim=1)
			
			# ===== your code here! =====
			# TODO:compute `pred_prev_image`, the DDPM estimate for the image at the
			# next timestep, which is slightly less noisy. Use the equation for
			# x_{t'} in the notebook.
			x0 = (image - torch.sqrt(1 - alpha_cumprod) * noise_est) / torch.sqrt(alpha_cumprod)

			pred_prev_image = (
				torch.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod) * x0 +
				torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * image
			)
			# ==== end of code ====

			# Add variance
			pred_prev_image = add_variance(predicted_variance, t, pred_prev_image, stage_1)
			image = pred_prev_image
		
		clean = image.cpu().detach().numpy()

	return clean, intermediate_images



def iterative_denoise_cfg(image, i_start, prompt_embeds, uncond_prompt_embeds, timesteps, alphas_cumprod, stage_1, scale=7):
	"""
	Iterative denoising with classifier free guidance: remove noise from the noisy image to obtain an estimate of the original image
	Args:
		image: noisy image of shape (B, 3, 64, 64)
		i_start: starting timestep index
		prompt_embeds: conditional prompt embeddings of shape (B, 77, 4096)
		uncond_prompt_embeds: unconditional prompt embeddings of shape (B, 77, 4096)
		timesteps: list of timesteps
		alphas_cumprod: cumulative alphas of shape (1000,)
		stage_1: DeepFloyd stage_1 model
		scale: CFG scale
	Returns:
		clean: denoised image of shape (B, 3, 64, 64)
	"""
	with torch.no_grad():
		for i in range(i_start, len(timesteps) - 1):
			# Get timesteps
			t = timesteps[i]
			prev_t = timesteps[i+1]

			# TODO: Get `alpha_cumprod`, `alpha_cumprod_prev`, `alpha`, `beta`
			# Feel free to copy code from part 3.4
			# ===== your code here! =====
			alpha_cumprod = alphas_cumprod[t].to(image.device)
			alpha_cumprod_prev = alphas_cumprod[prev_t].to(image.device)
			alpha = alpha_cumprod / alpha_cumprod_prev
			beta = 1 - alpha
			# ==== end of code ====

			# TODO: Get conditional noise estimate in model_output
			#  and unconditional noise estimate in uncond_model_output
			# ===== your code here! =====
			model_output = stage_1.unet(
				image, t, encoder_hidden_states=prompt_embeds, return_dict=False
			)[0]
			uncond_model_output = stage_1.unet(
				image, t, encoder_hidden_states=uncond_prompt_embeds, return_dict=False
			)[0]
			# ==== end of code ====

			# Split estimate into noise and variance estimate
			# model_output refers to the conditional noise estimate
			# uncond_output refers to the unconditional noise estimate
			noise_est, predicted_variance = torch.split(model_output, image.shape[1], dim=1)
			uncond_noise_est, _ = torch.split(uncond_model_output, image.shape[1], dim=1)
			
			# TODO: Compute the CFG noise estimate
			# Hint: Should only require one line of code
			# ===== your code here! =====
			eps = uncond_noise_est + scale * (noise_est - uncond_noise_est)
			# ==== end of code ====

			# TODO: Compute `pred_prev_image`, the next less noisy image using the CFG noise estimate.
			# HINT: Use the equation for x_{t'} in the notebook
			# ===== your code here! =====
			x0 = (image - torch.sqrt(1 - alpha_cumprod) * eps) / torch.sqrt(alpha_cumprod)

			pred_prev_image = (
				torch.sqrt(alpha_cumprod_prev) * beta / (1 - alpha_cumprod) * x0 +
				torch.sqrt(alpha) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod) * image
			)
			# ==== end of code ====

			# Add variance uses variance from the conditional prompt here
			pred_prev_image = add_variance(predicted_variance, t, pred_prev_image, stage_1)
			image = pred_prev_image

		clean = image.cpu().detach().numpy()

	return clean
