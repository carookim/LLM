from openai import OpenAI
import openai
import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

prompt = "안녕, 1만 입력해."

response = client.responses.create(
    model="gpt-4.1-mini",
    input=prompt
)

print(response.output_text)