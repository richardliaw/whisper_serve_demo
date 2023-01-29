from ray import serve

from fastapi import FastAPI, UploadFile
import whisper

app = FastAPI()


@serve.deployment(route_prefix="/hello", ray_actor_options={"num_gpus": 1})
@serve.ingress(app)
class WhisperDeployment:
    def __init__(self, language="english", model_size="base"):
        # there are no english models for large
        if model_size != 'large' and language == 'english':
            model = model_size + '.en'
        self.language = language
        self.audio_model = whisper.load_model(model_size, device="cuda")

    @app.post("/transcribe")
    def transcribe(
        self,
        audio_data: UploadFile,
    ):
        temp_file_name = "temp_file.mp3"
        with open(temp_file_name, "wb") as temp_file:
            temp_file.write(audio_data.file.read())
        print("Received audio data")
        if self.language == 'english':
            result = self.audio_model.transcribe(temp_file_name, language='english')
        else:
            result = self.audio_model.transcribe(temp_file_name)
        return result

serve_app = WhisperDeployment.bind()
handle = serve.run(serve_app)
import time

import requests
files = {'audio_data': open('some_recording.mp3','rb')}

start = time.time()
r = requests.post("http://localhost:8000/hello/transcribe", files=files)
print(r)
end = time.time()
print(r.content)
print("Total runtime", end - start)


# why not just serve.fastapi_wrapper(app, route_prefix="/hello")?