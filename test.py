import streamlit as st
# st.set_page_config(page_title="Voice Test")

# st.title("Voice Input Test")

# prompt = st.chat_input(
#     "Speak or type something...",
#     accept_audio=True
# )

# if prompt:
#     st.write("Text:", prompt.text)

#     if prompt.audio:
#         st.audio(prompt.audio)
#         st.success("Audio received successfully!")
import torch

print("PyTorch version:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print("GPU memory:",
      round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
      "GB")