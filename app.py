# importació de llibreries

import torch
import gradio as gr

from transformers import AutoProcessor, MusicgenForConditionalGeneration


# cpu o gpu

device = "cuda" if torch.cuda.is_available() else "cpu"


# model de generació de música

model_name = "facebook/musicgen-small"

processor = AutoProcessor.from_pretrained(model_name)

model = MusicgenForConditionalGeneration.from_pretrained(model_name)
model.to(device)


# funció de generació musical

def generar_musica(text):

    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt"
    ).to(device)

    audio_values = model.generate(
        **inputs,
        max_new_tokens=256
    )

    sampling_rate = model.config.audio_encoder.sampling_rate

    return (
        sampling_rate,
        audio_values[0, 0].cpu().numpy()
    )


# interfície Gradio

interficie = gr.Interface(
    fn=generar_musica,
    inputs=gr.Textbox(
        lines=2,
        label="Descripció de la música"
    ),
    outputs=gr.Audio(
        label="Música generada"
    ),
    title="Generador de música amb MusicGen",
    description="Introdueix una descripció i genera una peça musical."
)


# executar aplicació

interficie.launch()
