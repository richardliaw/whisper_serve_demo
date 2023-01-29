from json import load
import os
import tempfile
from ray import serve
import pydub
from asyncio import as_completed

from fastapi import FastAPI, UploadFile, Form
import whisper

app = FastAPI()

@serve.deployment()
@serve.ingress(app)
class AudioIngress:
    def __init__(self, whispers):
        self.whispers = whispers
        print("Instantiated.")

    @app.post("/transcribe")
    async def transcribe(
        self,
        audio_data: UploadFile,
    ):
        print("Received audio data")
        filepath = "temp_file.mp3"
        with open(filepath, "wb") as temp_file:
            temp_file.write(audio_data.file.read())

        from pydub import AudioSegment

        # Load the audio file
        audio = AudioSegment.from_mp3(filepath)

        # Define the duration of each segment (in milliseconds)
        segment_duration = 10000

        # Get the number of segments
        num_segments = len(audio) // segment_duration
        print(f"Creating {num_segments} segments")

        # Initialize an empty list to store the segments
        segments = []

        # Iterate through the segments
        for i in range(num_segments):
            # Extract the segment
            segment = audio[i*segment_duration:(i+1)*segment_duration]
            # Append the segment to the list
            segments.append(self.whispers.transcribe.remote(segment, i))

        queued_tasks = []
        for coro in as_completed(segments):
            queued_task = await coro
            queued_tasks.append(queued_task)

        print(queued_tasks)

        finished = {}
        for coro in as_completed(queued_tasks):
            earliest_result, index = await coro
            finished[index] = earliest_result

        print(finished)
        return finished


@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 100,
        "target_num_ongoing_requests_per_replica": 1,
        "upscale_delay_s": 2
    },
    ray_actor_options={"num_gpus": 0.1})
class WhisperDeployment:
    def __init__(self, language="english", model_size="base"):
        # there are no english models for large
        if model_size != 'large' and language == 'english':
            model = model_size + '.en'
        self.language = language
        self.audio_model = whisper.load_model(model_size, device="cuda")
        # self.audio_model = load_model_multigpu(model_size)

    async def transcribe(self, audio_segment, index):
        with tempfile.NamedTemporaryFile() as tmpfile:
            audio_segment.export(str(tmpfile), format="mp3")
            print("Received audio segment")
            if self.language == 'english':
                result = self.audio_model.transcribe(str(tmpfile), language='english')
            else:
                result = self.audio_model.transcribe(str(tmpfile))
        return result, index

serve_app = AudioIngress.bind(WhisperDeployment.bind())
handle = serve.run(serve_app)
import time
# time.sleep(10000)

import requests
files = {'audio_data': open('some_recording.mp3','rb')}

start = time.time()
r = requests.post("http://localhost:8000/transcribe", files=files)
print(r)
end = time.time()
print(r.content.decode())
print("Total runtime", end - start)
serve.shutdown()
