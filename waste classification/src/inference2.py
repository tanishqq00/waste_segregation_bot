import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
LABELS = ["Electronics", "Inorganic", "Organic", "Metal", "Others"]
DEVICE = "cpu"
MAX_SIDE = 80

def parse_prediction(text: str) -> str:
    cleaned = text.strip().lower()
    for label in LABELS:
        if label.lower() in cleaned:
            return label
    print(f"Warning: Could not find label in response '{text}'. Defaulting to Others.")
    return "Others"


def classify_image(processor, model, image, device):
    """Classify a single PIL image using SmolVLM."""
    if image is None:
        return "Others"

    prompt = (
        "You are a waste classification expert. Identify the main object and "
        "classify it into one of these categories: Electronics, Inorganic, Organic, Metal, or Others.\n\n"
        "Answer with only the category name."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    try:
        text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(device)

        # Ensure float precision
        inputs = {k: v.to(torch.float32) if v.dtype == torch.float16 else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=processor.tokenizer.eos_token_id,
            )

        response = processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        #print(f"Raw response: '{response}'")
        return parse_prediction(response)

    except Exception as e:
        print(f"Error during classification: {e}")
        return "Others"


def capture_single_image():
    """Automatically capture a single frame from webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    # Allow a short warm-up
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Failed to capture frame.")
        return None

    return frame


def main():
    print(f"Loading model '{MODEL_ID}' on {DEVICE.upper()}...")
    model = AutoModelForVision2Seq.from_pretrained(MODEL_ID, torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    print("Model and processor loaded.\n")

    frame = capture_single_image()
    if frame is None:
        return

    # Convert and resize
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize((MAX_SIDE, MAX_SIDE))

    # Run classification
    prediction = classify_image(processor, model, pil_img, DEVICE)

    # Display result
    print(f"\nPredicted Category: {prediction}")



if __name__ == "__main__":
    main()
