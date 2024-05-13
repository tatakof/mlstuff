import os
import requests
from dsp import LM
from dspy import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot
api_key=os.environ.get("ANTHROPIC_API_KEY")

class Claude(LM):
    def __init__(self, model, api_key, **kwargs):
        self.model = model
        self.api_key = api_key
        self.provider = "default"
        self.kwargs = kwargs
        self.history = []
        self.base_url = "https://api.anthropic.com/v1/messages"

    def basic_request(self, prompt: str, **kwargs):
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": "messages-2023-12-15",
            "content-type": "application/json"
        }
        print(f"\n request's headers: \n --------- \n {headers}")

        data = {
            **kwargs,
            "model": self.model,
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        print(f"\n request's data: \n ---------- \n {data}")

        response = requests.post(self.base_url, headers=headers, json=data)
        print("\n anthropic response to request: \n --------- \n {response}")
        response = response.json()

        self.history.append({
            "prompt": prompt, 
            "response": response,
            "kwargs": kwargs,
        })

        return response

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        response = self.basic_request(prompt, **kwargs)
        completions = [result["text"] for result in response["content"]]

        return completions


"""
This is an example of a proper request to Anthropic: 
-------------------


curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-3-opus-20240229",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, world"}
  ]
}'

"""

#model = "claude-3-opus-20240229"
#model = "claude-3-sonnet-20240229"
model = "claude-3-haiku-20240307"
kwargs = { "temperature": 1,
    "max_tokens": 1000
}
prompt = "hey can you explain me how syntactic annotation works?"

claude = Claude(model, api_key = api_key, **kwargs)
claude.basic_request(prompt=prompt)
claude(prompt=prompt)
dspy.settings.configure(lm=claude)
predict = dspy.Predict('question -> answer')
print(predict)

result = predict(question="What is the morphological annotation of the following sentence?\n Ær Cristes geflæscnesse lx wintra Gaius Iulius se casere ærest Romana Brytenland gesohte")
print(result)

claude.history
