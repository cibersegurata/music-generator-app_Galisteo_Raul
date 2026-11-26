# importació de llibreries

import torch
import streamlit as st

from transformers import AutoProcessor, MusicgenForConditionalGeneration


# cpu o gpu

device = "cuda" if torch.cuda.is_available() else "cpu"


# model de generació de música

model_name = "facebook/musicgen-small"

processor = AutoProcessor.from_pretrained(model_name)

model = MusicgenForConditionalGeneration.from_pretrained(
    model_name,
    low_cpu_mem_usage=False
)

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


# interfície Streamlit

st.title("Generador de música amb MusicGen")

st.write(
    "Introdueix una descripció i genera una peça musical."
)

text = st.text_input(
    "Descripció de la música"
)

if st.button("Generar música"):

    if text.strip():

        with st.spinner("Generant música..."):
            sampling_rate, audio_data = generar_musica(text)

        st.audio(
            audio_data,
            sample_rate=sampling_rate
        )

    else:
        st.warning("Introdueix una descripció.")
