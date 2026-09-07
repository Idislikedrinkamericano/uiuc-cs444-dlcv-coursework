import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
import mediapy as media
from PIL import Image

# seed_everything
def seed_everything(seed):
	torch.cuda.manual_seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# add_variance
def add_variance(predicted_variance, t, image, stage_1):
	'''
	Args:
		predicted_variance : (1, 3, 64, 64) tensor, last three channels of the UNet output
		t: scale tensor indicating timestep
		image : (1, 3, 64, 64) tensor, noisy image
		stage_1: DeepFloyd stage 1 model

	Returns:
		(1, 3, 64, 64) tensor, image with the correct amount of variance added
	'''
	# Add learned variance
	variance = stage_1.scheduler._get_variance(t, predicted_variance=predicted_variance)
	variance_noise = torch.randn_like(image)
	variance = torch.exp(0.5 * variance) * variance_noise
	return image + variance

# process_pil_im
# @title Function to Process Images

def process_pil_im(img):
	'''
	Transform a PIL image
	'''

	# Convert to RGB
	img = img.convert('RGB')

	# Define the transform to resize, convert to tensor, and normalize to [-1, 1]
	transform = transforms.Compose([
		transforms.Resize(64),               # Resize shortest side to 64
		transforms.CenterCrop(64),             # Center crop
		transforms.ToTensor(),               # Convert image to PyTorch tensor with range [0, 1]
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1, 1]
	])

	# Apply the transformations and add batch dim
	img = transform(img)[None]

	# Show image
	print("Processed image")
	media.show_image(img[0].permute(1,2,0) / 2 + 0.5)

	return img

# upsample
def upsample(image64, prompt_embeds, stage_2, prompt_embeds_dict):
	"""
	Args:
		image64 : torch tensor of size (1, 3, 64, 64) representing the image
		prompt_embeds : torch tensor of size (1, 77, 4096), prompt embedding

	Returns:
		image256 : torch tensor of size (1, 3, 256, 256), upsampled image
	"""

	image256 = stage_2(
		image=image64,
		num_inference_steps=30,
		prompt_embeds=prompt_embeds,
		negative_prompt_embeds=prompt_embeds_dict[''],
		output_type="pt",
	).images.cpu()

	return image256
