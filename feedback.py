import brain
import config
from dotenv import load_dotenv
from google import genai
from google.genai import types
from httpx import ReadError, ConnectTimeout, ConnectError
import json
import os
import pickle
import processing
from requests import exceptions
from rich.console import Console
from time import sleep


with open(config.system_prompt_file, 'r') as f:
    system_prompt = f.read()
    system_prompt = system_prompt.strip()

with open(config.simulations_file, 'w+') as f:
    json.dump([], f, indent = 2)

with open(config.context_file, 'wb') as f:
    pickle.dump([], f)

if (config.operators):
    with open("operators.txt", "r") as f:
        operators = f.read()
        operators = operators.strip()

    system_prompt = system_prompt.replace("{Operators}", operators)
    system_prompt = system_prompt.strip()

load_dotenv()
console = Console()
brain_session = brain.login()
genai_client = genai.Client(
    api_key = os.getenv('gemini_api_keys').split(',')[1]
)

context = []


model_config = types.GenerateContentConfig(
    thinking_config = types.ThinkingConfig(
        thinking_budget = config.thinking_budget
    ),
    temperature = config.temperature,
    response_mime_type = 'application/json',
    response_schema = types.Schema(
        type = types.Type.OBJECT,
        description = config.schema_description,
        required = list(config.schema.keys()),
        properties = {
            'Alpha Expression': types.Schema(
                type = types.Type.STRING,
                description = config.schema['Alpha Expression'],
            ),
            'Reasoning': types.Schema(
                type = types.Type.STRING,
                description = config.schema['Reasoning']
            )
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
        response = genai_client.models.generate_content(
            model = config.model,
            contents = context,
            config = model_config
        )

        resp = response.text
        return json.loads(resp)
    
    def get_context(i, model_output):

        model_context = f"""
Iteration #{i + 1}
Alpha Expression:
{model_output['Alpha Expression']}

Reasoning:
{model_output['Reasoning']}
"""
        return model_context.strip()


class User:
    def get_context(simulation_result, score):
        insample = simulation_result['is']
        train = simulation_result['train']
        checks = insample['checks']

        is_sub_universe_sharpe = [check['value'] for check in checks if check['name'] == 'LOW_SUB_UNIVERSE_SHARPE'][0]
        is_sub_universe_sharpe_limit = [check['limit'] for check in checks if check['name'] == 'LOW_SUB_UNIVERSE_SHARPE'][0]

        user_context = f"""
Simulation Results:
Test Period Score: {score}
Train Period Sharpe: {train['sharpe']}
Train Period Fitness: {train['fitness']}
Train Period Turnover: {round(100 * train['turnover'], 2)}%
Sub Universe Robustness: {round(is_sub_universe_sharpe / min(0.1, is_sub_universe_sharpe_limit), 2)}
"""

        if (checks[4]['result'] == 'FAIL'):
            if (checks[4].get('value')):
                user_context += f'Weight Concentration {round(checks[4]['value'] * 100, 2)}% is above cutoff of {round(checks[4]['limit'] * 100, 2)}%.\n'
            else:
                user_context += 'Weight is too strongly concentrated or too few instruments are assigned weight.\n'

        # if (checks[5]['result'] == 'FAIL'):
            # user_context += f'In Sample Sub Universe Sharpe {checks[5]['value']} is not above {checks[5]['limit']}.\n'

        # if (checks[6]['result'] == 'UNITS'):
        #     user_context += checks[6]['message'].replace('; ', '\n')
        #     user_context += '\n'

        return user_context.strip()
    
    def process_output(model_output):

        payload = {
            'type': 'REGULAR',
            'settings': config.simulation_settings,
            'regular': model_output['Alpha Expression']
        }

        return payload

    def save_iteration(context, alpha):

        with open(config.simulations_file, 'r+') as f:
            data = json.load(f)
            data.append(alpha)
            f.seek(0)
            json.dump(data, f, indent = 2)

        with open(config.context_file, 'wb') as f:
            pickle.dump(context, f)


context.append(
    types.Content(
        role = 'user',
        parts = [
            types.Part.from_text(text = config.initial_prompt)
        ]
    )
)


console.print(f'Model: {config.model}', style = 'purple')
console.print(f'Temperature: {config.temperature}', style = 'purple')
console.print(f'System Prompt File: {config.system_prompt_file}', style = 'purple')


console.print(f'{config.initial_prompt}', style = 'green')


for i in range(config.max_iterations):

    while True:
        try:
            model_output = Model.get_output(context)
            break
        except (ReadError, genai.errors.ServerError, ConnectTimeout, ConnectError) as e:
            console.print(f"Model.get_output: {e}", style = "red")
            sleep(10)

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
        except (exceptions.ConnectionError, exceptions.JSONDecodeError) as e:
            console.print(f"brain.Alpha.simulate: {e}", style = "red")
            sleep(10)

    pnl_data = brain.Alpha.pnl(brain_session, alpha_id)
    processing.pnl_chart(pnl_data)

    simulation_results = brain.Alpha.simulation_result(brain_session, alpha_id)
    score = processing.score(simulation_results)


    user_context = User.get_context(simulation_results, score)

    # with open('pnl_chart.png','rb') as f:
    #     image_bytes = f.read()

    # pnl_chart = types.Part.from_bytes(data = image_bytes, mime_type = "image/png")

    context.append(
        types.Content(
            role = 'user',
            parts = [
                types.Part.from_text(text = user_context)
            ]
        )
    )
    console.print(user_context, style = 'green')

    User.save_iteration(context, simulation_results)

    token_count = Model.count_tokens(context)
    console.print(f'Token Count: {token_count}', style = 'purple')

    print()