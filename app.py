import gradio as gr
import cv2
import numpy as np
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from utils import enhance_lighting

# Ensure embeddings folder exists
os.makedirs("embeddings", exist_ok=True)

# Load InsightFace model
print("[INFO] Loading InsightFace model...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=-1, det_size=(640, 640))
print("[INFO] Model Loaded Successfully!")


def load_embeddings():
    names = []
    embeddings = []

    for file in os.listdir("embeddings"):
        if file.endswith(".npy"):
            emb = np.load(os.path.join("embeddings", file))
            embeddings.append(emb)
            names.append(file.replace(".npy", ""))

    if len(embeddings) == 0:
        return np.array([]), []

    return np.vstack(embeddings), names


def register_person(name, images):
    if not name:
        return "❌ Please enter a name."

    if not images:
        return "❌ Upload at least 1 face image."

    emb_list = []

    for img in images:
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces = face_app.get(frame)

        if len(faces) == 0:
            continue

        emb_list.append(faces[0].embedding)

    if len(emb_list) == 0:
        return "❌ No face detected. Try clearer images."

    avg_emb = np.mean(emb_list, axis=0)
    np.save(f"embeddings/{name}.npy", avg_emb)

    return f"✅ Successfully registered: {name}"


def recognize_frame(frame):
    if frame is None:
        return frame, "No frame received"

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame_bgr = enhance_lighting(frame_bgr)

    faces = face_app.get(frame_bgr)
    known_embs, names = load_embeddings()

    annotated = frame_bgr.copy()

    if len(known_embs) == 0:
        return frame, "⚠ No registered faces found"

    label = "No face detected"

    for f in faces:
        box = f.bbox.astype(int)
        emb = f.embedding.reshape(1, -1)

        sims = cosine_similarity(emb, known_embs)[0]
        idx = np.argmax(sims)
        score = sims[idx]

        name = names[idx] if score > 0.45 else "Unknown"
        label = f"{name} ({score:.2f})"

        cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        cv2.putText(annotated, label, (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return annotated_rgb, label


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🔥 Real-Time Face Recognition (Mask, Glasses, Hat Compatible)")

    with gr.Tab("➕ Register Face"):
        name = gr.Textbox(label="Enter Name")
        imgs = gr.Image(type="numpy", source="upload", label="Upload Face Images", multiple=True)
        out = gr.Textbox()
        btn = gr.Button("Register Face")
        btn.click(register_person, inputs=[name, imgs], outputs=out)

    with gr.Tab("🎥 Live Recognition"):
        webcam = gr.Image(source="webcam", streaming=True, type="numpy", label="Webcam Stream")
        output_img = gr.Image()
        status = gr.Textbox()
        webcam.stream(recognize_frame, inputs=webcam, outputs=[output_img, status])


# IMPORTANT FIX FOR RENDER.COM
port = int(os.getenv("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)
