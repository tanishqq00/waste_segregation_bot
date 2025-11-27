import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from collections import deque, Counter
import matplotlib.pyplot as plt
import numpy as np

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
LABELS = ["Electronics", "Inorganic", "Organic", "Metal", "Others"]

HISTORY_SIZE = 5
PROCESS_EVERY_N_FRAMES = 15
prediction_history = deque(maxlen=HISTORY_SIZE)

def imshow_available():
    try:
        cv2.namedWindow("Test")
        cv2.destroyWindow("Test")
        return True
    except cv2.error:
        return False


USE_IMSHOW = imshow_available()
if not USE_IMSHOW:
    print("OpenCV GUI not available — falling back to Matplotlib display.")


def parse_prediction(text: str) -> str:
    cleaned = text.strip().lower()
    for label in LABELS:
        if label.lower() in cleaned:
            return label
    print(f"Warning: Could not find label in response '{text}'. Defaulting to Others.")
    return "Others"


def classify_waste_with_smolvlm(processor, model, image, device):
    """
    Classify waste using SmolVLM with a lightweight few-shot prompt.
    """
    if image is None:
        return "Others"

    prompt = (
        "You are a waste classification expert. Identify the main object and "
        "classify it into one of these categories: Electronics, Inorganic, Organic, Metal, or Others.\n\n"
        "Definitions:\n"
        "- Electronics: Devices with circuits/batteries (phones, remotes, chargers, etc).\n"
        "- Inorganic: Man-made, non-biodegradable (plastic, glass, styrofoam).\n"
        "- Organic: Biodegradable (paper, food, leaves).\n"
        "- Metal: Mostly metallic items (can, foil, thermos).\n"
        "- Others: Anything else.\n"
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

        print(f"Raw response: '{response}'")
        return parse_prediction(response)

    except Exception as e:
        print(f"Error during classification: {e}")
        return "Others"


def main():
    device = "cpu"
    print(f"Loading model '{MODEL_ID}' on {device.upper()}...")

    try:
        model = AutoModelForVision2Seq.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32
        ).to(device)
        model.eval()
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        print(" Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("📷 Camera ready. Press 'q' to quit.")
    stable_prediction = "Initializing..."
    frame_counter = 0

    if not USE_IMSHOW:
        plt.ion()
        fig, ax = plt.subplots()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to grab frame.")
                break

            h, w, _ = frame.shape
            roi_size = int(min(h, w) * 0.6)
            x_start = (w - roi_size) // 2
            y_start = (h - roi_size) // 2
            x_end = x_start + roi_size
            y_end = y_start + roi_size

            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            frame_counter += 1

            if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                roi_frame = frame[y_start:y_end, x_start:x_end]
                if roi_frame.size == 0:
                    continue

                img_rgb = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
                max_side = 224  # SmolVLM's safe max size
                pil_img = Image.fromarray(img_rgb)
                pil_img = pil_img.resize((max_side, max_side))

                current_pred = classify_waste_with_smolvlm(processor, model, pil_img, device)
                prediction_history.append(current_pred)

                if prediction_history:
                    stable_prediction = Counter(prediction_history).most_common(1)[0][0]

                print(f"Current: {current_pred} → Stable: {stable_prediction}")
                print("-" * 50)

            cv2.rectangle(frame, (5, 5), (450, 60), (0, 0, 0), -1)
            cv2.putText(frame, f"Prediction: {stable_prediction}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if USE_IMSHOW:
                cv2.imshow("SmolVLM Waste Classifier", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                ax.clear()
                ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ax.set_title(f"Prediction: {stable_prediction}")
                ax.axis("off")
                plt.pause(0.001)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        if USE_IMSHOW:
            cv2.destroyAllWindows()
        else:
            plt.close(fig)
        print(" Resources released.")


if __name__ == "__main__":
    main()
