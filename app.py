import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("best_model.keras")

CLASSES   = ["papaya", "persimmon", "pomelo"]
LABELS_VN = {"papaya": "Đu đủ", "persimmon": "Hồng", "pomelo": "Bưởi"}
IMG_SIZE  = (224, 224)

def predict(image):
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, 0)
    probs = model.predict(arr, verbose=0)[0]
    return {LABELS_VN[c]: float(p) for c, p in zip(CLASSES, probs)}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload ảnh trái cây"),
    outputs=gr.Label(num_top_classes=3, label="Kết quả phân loại"),
    title="Nhận dạng Đu đủ / Hồng / Bưởi",
    description="Upload ảnh trái cây, mô hình AI sẽ phân loại tự động.",
    flagging_mode="never",
)

demo.launch()
