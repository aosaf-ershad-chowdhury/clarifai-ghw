from clarifai.client.model import Model

prompt = "Build me a simple agent with Clarifai Python SDK that can answer questions about the weather."

model_url="https://clarifai.com/qwen/qwenLM/models/Qwen3-14B"
# This api key wont work btw :)
model_prediction = Model(url=model_url, pat="11d540cd9603427a82ee0572e86c3521").predict_by_bytes(prompt.encode())

print(model_prediction.outputs[0].data.text.raw)
