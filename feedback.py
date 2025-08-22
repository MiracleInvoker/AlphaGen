import brain
import config
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import os
import pickle
import processing
from random import randint
from rich.console import Console
from time import sleep, strftime
from utils import terminal


load_dotenv()
console = Console()
gemini_api_keys = os.getenv('gemini_api_keys').split(',')
output_file = strftime('%d%m%Y-%H%M%S')
simulations_file = f"simulations/{output_file}.json"
contexts_file = f"contexts/{output_file}.pkl"
gemini_api_key_id = randint(0, len(gemini_api_keys) - 1)
context = []

brain_session = brain.login()
genai_client = genai.Client(
    api_key = gemini_api_keys[gemini_api_key_id]
)

os.makedirs("simulations", exist_ok = True)
os.makedirs("contexts", exist_ok = True)

with open(config.system_prompt_file, 'r') as f:
    system_prompt = f.read()

with open(config.operators_file, 'r') as f:
    operators = f.read()
    operators = operators.strip()

system_prompt = system_prompt.replace("{Operators}", operators)
system_prompt = system_prompt.strip()

with open(simulations_file, 'w+') as f:
    json.dump([], f, indent = 2)

with open(contexts_file, 'wb') as f:
    pickle.dump(context, f)


model_config = types.GenerateContentConfig(
    thinking_config = types.ThinkingConfig(
        thinking_budget = config.thinking_budget
    ),
    temperature = config.temperature,
    response_mime_type = 'application/json',
    response_schema = types.Schema(
        type = types.Type.OBJECT,
        description = config.structured_output_schema_description,
        required = list(config.structured_output_schema.keys()),
        properties = {
            key: types.Schema(
                type = types.Type.STRING,
                description = desc
            )
            for key, desc in config.structured_output_schema.items()
        },
    ),
    system_instruction = [
        types.Part.from_text(text = system_prompt),
    ]
)


class Model:
    def count_tokens(context):
        resp = genai_client.models.count_tokens(
            model = config.model,
            contents = context
        )

        return resp.total_tokens

    def get_output(context):

        trial = 0

        while True:

            trial += 1

            terminal.clear_line()
            console.print(f"Attempt #{trial} | Retrieving Model Output...", end = '', style = 'yellow')

            response = genai_client.models.generate_content(
                model = config.model,
                contents = context,
                config = model_config
            )

            if (response.parsed):
                break

            sleep(5)

        terminal.clear_line()
        console.print(f"Attempt #{trial} | Model Output Retrieved.", style = 'yellow')

        return response.parsed

    def get_context(i, model_output):

        lines = [f"Iteration #{i + 1}"]

        for field in config.structured_output_schema:
            lines.append(f"{field}:")
            lines.append(model_output.get(field))

        model_context = "\n".join(lines)

        return model_context


class User:
    def get_context(simulation_result):

        kpis = processing.get_kpis(simulation_result)
        user_context = []

        for kpi in kpis.keys():
            metric = kpis[kpi]

            if (kpi == "SUBMITTABLE"): pass
            
            if (kpi == "WARNINGS"):
                user_context += metric                
                break
            
            txt = f"{kpi}: {metric[0]}"
            
            if (len(metric) == 2):
                if (metric[0] < metric[1]):
                    txt += f" is less than {metric[1]}"
                    
            if (len(metric) == 3):
                if (metric[0] < metric[1]):
                    txt += f" is less than {metric[1]}"
                    
                if (metric[0] > metric[2]):
                    txt += f" is more than {metric[2]}"

            user_context.append(txt)
            
        return kpis["SUBMITTABLE"], "\n".join(user_context)

    def process_output(model_output):

        payload = {
            'type': 'REGULAR',
            'settings': config.simulation_settings,
            'regular': model_output['Alpha Expression']
        }

        return payload

    def save_iteration(context, alpha):

        with open(simulations_file, 'r+') as f:
            data = json.load(f)
            data.append(alpha)
            f.seek(0)
            json.dump(data, f, indent = 2)

        with open(contexts_file, 'wb') as f:
            pickle.dump(context, f)


initial_prompt = config.initial_prompt
initial_prompt += "\n\nData Field Context:\n```\n"

for data_field in config.data_fields:
    data_field_desc = brain.data_field(brain_session, data_field)
    initial_prompt += f"{data_field}: {data_field_desc}\n"

initial_prompt += "```"

context.append(
    types.Content(
        role = 'user',
        parts = [
            types.Part.from_text(text = initial_prompt)
        ]
    )
)


console.print(f"Model: {config.model}", style = 'purple')
console.print(f"Temperature: {config.temperature}", style = 'purple')
console.print(f"System Prompt File: {config.system_prompt_file}", style = 'purple')
console.print(f"Gemini API Key ID: [{gemini_api_key_id}]", style = 'purple')


console.print(initial_prompt, style = 'green')


for i in range(config.max_iterations):

    while True:

        try:
            model_output = Model.get_output(context)
            break

        except Exception as e:
            error = e.args[0]
            print()

            if ("RESOURCE_EXHAUSTED" in error):

                gemini_api_keys.pop(gemini_api_key_id)
                console.print(f"Gemini API Key ID [{gemini_api_key_id}] | 429 RESOURCE_EXHAUSTED", style = 'red')

            gemini_api_key_id = randint(0, len(gemini_api_keys) - 1)

            console.print(f"Changing GEMINI API Key ID... [{gemini_api_key_id}]", style = 'yellow')

            genai_client = genai.Client(
                api_key = gemini_api_keys[gemini_api_key_id]
            )

            sleep(5)

    model_context = Model.get_context(i, model_output)
    simulation_data = User.process_output(model_output)

    context.append(
        types.Content(
            role = 'model',
            parts = [
                types.Part.from_text(text = model_context)
            ]
        )
    )
    console.print(model_context, style = 'cyan')

    while True:
        try:
            alpha_id = brain.Alpha.simulate(brain_session, simulation_data)
            break
        except Exception as e:
            console.print(f"brain.Alpha.simulate: {e}", style = 'red')
            sleep(5)

    pnl_data = brain.Alpha.pnl(brain_session, alpha_id)
    processing.pnl_chart(pnl_data)

    while True:
        try:
            simulation_result = brain.Alpha.simulation_result(brain_session, alpha_id)
            break
        except Exception as e:
            console.print(f"brain.Alpha.simulation_result: {e}", style = 'red')
            sleep(5)

    is_submittable, user_context = User.get_context(simulation_result)

    context.append(
        types.Content(
            role = 'user',
            parts = [
                types.Part.from_text(text = user_context)
            ]
        )
    )
    console.print(user_context, style = 'green')

    User.save_iteration(context, simulation_result)

    try:
        token_count = Model.count_tokens(context)
        console.print(f"Token Count: {token_count}", style = 'purple')

    except Exception as e:
        console.print(f"Model.count_tokens: {e}", style = 'red')

    print()

    if (is_submittable):
        success_text = r"""
     _    _       _                               
    / \  | |_ __ | |__   __ _                     
   / _ \ | | '_ \| '_ \ / _` |                    
  / ___ \| | |_) | | | | (_| |                    
 /_/___\_\_| .__/|_| |_|\__,_|    _           _ _ 
  / ___| __|_| __   ___ _ __ __ _| |_ ___  __| | |
 | |  _ / _ \ '_ \ / _ \ '__/ _` | __/ _ \/ _` | |
 | |_| |  __/ | | |  __/ | | (_| | ||  __/ (_| |_|
  \____|\___|_| |_|\___|_|  \__,_|\__\___|\__,_(_)
                                                  
"""
        console.print(success_text, style = 'green')
        break