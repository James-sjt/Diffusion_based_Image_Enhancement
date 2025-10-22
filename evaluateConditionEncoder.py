import numpy as np
from huggingface_hub import hf_hub_download
from latentDiffusion import CondImageEncoder
import torch
import matplotlib.pyplot as plt
from diffusionDataset import ImageDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
unet_path = hf_hub_download(repo_id="James0323/enhanceImg", filename="unet_best.pth")
# Hook function to capture feature maps
def hook_fn(module, input, output):
    # Capture output and store it in a dictionary
    output = output[0].cpu().detach().numpy()
    return output


def visualize_input_and_feature_maps(input_img, feature_maps):
    # Create a grid of subplots to display the input image and feature maps
    fig, axes = plt.subplots(1, 7, figsize=(15, 10))

    # Display the input image
    input_img = input_img.squeeze(0).to("cpu").detach()
    axes[0].imshow(input_img.permute(1, 2, 0), cmap='gray')  # Convert (C, H, W) to (H, W, C)
    axes[0].set_title("Input Image")
    for i, feature_map in enumerate(feature_maps['layer1'][: 6]):
        # If feature_map is a 2D array (height, width), show it directly
        if len(feature_map.shape) == 2:
            feature_map = np.expand_dims(feature_map, -1)
            axes[i + 1].imshow(feature_map, cmap='gray')
        else:
            # If feature_map is 3D, select the first channel (filter)
            axes[i + 1].imshow(feature_map, cmap='gray')  # Show first filter

        axes[i + 1].set_title(f"Feature Map {i + 1}")

    plt.show()

# Register hooks to capture features from each layer
def register_hooks(model):
    hooks = []
    feature_maps = {}

    def gradient_weight(f_c):
        # Simple Sobel-based gradient magnitude
        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        sobel_y = sobel_x.transpose(2, 3)
        gx = F.conv2d(f_c, sobel_x.expand(f_c.size(1), 1, 3, 3), padding=1, groups=f_c.size(1))
        gy = F.conv2d(f_c, sobel_y.expand(f_c.size(1), 1, 3, 3), padding=1, groups=f_c.size(1))
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        w = grad_mag / (grad_mag.mean(dim=(2, 3), keepdim=True) + 1e-6)
        return f_c * w

    def hook_fn_layer1(module, input, output):
        print(output.shape)
        output = gradient_weight(output)
        feature_maps['layer1'] = output[0].cpu().detach().numpy()

    def hook_fn_layer2(module, input, output):
        output = gradient_weight(output)
        feature_maps['layer2'] = output[0].cpu().detach().numpy()

    def hook_fn_layer3(module, input, output):
        output = gradient_weight(output)
        feature_maps['layer3'] = output[0].cpu().detach().numpy()

    def hook_fn_layer4(module, input, output):
        output = gradient_weight(output)
        feature_maps['layer4'] = output[0].cpu().detach().numpy()

    # Attach hooks to layers
    hooks.append(model.proj1.register_forward_hook(hook_fn_layer1))
    hooks.append(model.proj2.register_forward_hook(hook_fn_layer2))
    hooks.append(model.proj3.register_forward_hook(hook_fn_layer3))
    hooks.append(model.proj4.register_forward_hook(hook_fn_layer4))

    return hooks, feature_maps

conditionEncoder = CondImageEncoder(cond_dim=512).to(device)

# Load the checkpoint
checkpoint = torch.load(unet_path)
checkpoint_state_dict = checkpoint  # The state dict is the full checkpoint, not a sub-key like 'model_state_dict'
new_state_dict = {}
for k, v in checkpoint_state_dict.items():
    new_key = k.replace('cond_encoder.', '')  # Remove the 'cond_encoder.' prefix
    new_state_dict[new_key] = v

conditionEncoder.load_state_dict(new_state_dict, strict=False)

conditionEncoder.eval()
for param in conditionEncoder.parameters():
    param.requires_grad = False

hooks, feature_maps = register_hooks(conditionEncoder)
dataset_valid = ImageDataset('train', device, True)
dl_val = DataLoader(dataset_valid, batch_size=1, shuffle=True)

for idx, (img, _, mask) in enumerate(dl_val):
    img = img.to(device)
    img = img * 2 - 1
    conditionEncoder(img)
    visualize_input_and_feature_maps(img, feature_maps)
    break

for hook in hooks:
    hook.remove()

