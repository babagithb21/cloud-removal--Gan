from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from generator import Generator

app = Flask(__name__)
generator = Generator()
generator.load_state_dict(torch.load('generator.pth', map_location=torch.device('cpu')))
generator.eval()

preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)

    with torch.no_grad():
        generated_img = generator(img_tensor).squeeze().permute(1, 2, 0).numpy()
        generated_img = (generated_img * 255).astype('uint8')

    output_image = Image.fromarray(generated_img)
    output_path = 'static/generated_image.png'
    os.makedirs('static', exist_ok=True)
    output_image.save(output_path)

    return jsonify({'output_img_url': output_path})

if __name__ == '__main__':
    app.run(debug=True)
