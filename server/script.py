import torch
from PIL import Image
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer
import torchvision.transforms as transforms
import os

# Assuming you have the necessary components and the MinImagen model architecture defined
# Adjust this part to match the actual MinImagen model loading code
class MinImagenModel(torch.nn.Module):
    # Define the model architecture here
    def __init__(self):
        super(MinImagenModel, self).__init__()
        # Add model layers
        pass

    def forward(self, text_embedding):
        # Define forward pass
        pass

def load_model(model_path):
    model = MinImagenModel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_image(model, text, tokenizer, text_encoder, device):
    # Tokenize and encode the text
    inputs = tokenizer(text, return_tensors='pt')
    text_embeddings = text_encoder(**inputs).last_hidden_state
    
    # Generate image
    with torch.no_grad():
        generated_image = model(text_embeddings.to(device))

    # Post-process image tensor to convert it to a PIL image
    generated_image = generated_image.squeeze().cpu().numpy()
    generated_image = (generated_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(generated_image)
    
    return pil_image

def save_image(image, save_path):
    image.save(save_path)

def main():
    # Paths
    model_path = './model/minimagen_model.pth'  # Path to your trained model
    save_path = './output/generated_image.png'  # Path to save the generated image

    # Text input
    text = "A beautiful landscape with mountains and a river"

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(model_path).to(device)

    # Load tokenizer and text encoder (e.g., using CLIP)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

    # Generate image
    generated_image = generate_image(model, text, tokenizer, text_encoder, device)

    # Save generated image
    save_image(generated_image, save_path)
    print(f"Generated image saved at {save_path}")

if __name__ == "__main__":
    main()
