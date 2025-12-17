# stable-diffusion-xl-base-1.0
# black-forest-labs/FLUX.1-dev <- 사용 횟수제한 있음, 유료
import os
from dotenv import load_dotenv
load_dotenv()

import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="together",
    api_key=os.environ["HF_TOKEN"],
)

# output is a PIL.Image object
image = client.text_to_image(
    "draw a angry dinosaur",
    model="black-forest-labs/FLUX.1-dev",
)

image.show()