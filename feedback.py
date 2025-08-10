import brain
import config
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import os
import pickle
import processing
from rich.console import Console
from time import sleep, strftime


load_dotenv()
console = Console()
gemini_api_keys = os.getenv('gemini_api_keys').split(',')
output_file = strftime('%d%m%Y-%H%M%S')
simulations_file = f"simulations/{output_file}.json"
contexts_file = f"contexts/{output_file}.pkl"
context = []

brain_session = brain.login()
genai_client = genai.Client(
    api_key = gemini_api_keys[0]
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
        response = genai_client.models.generate_content(
            model = config.model,
            contents = context,
            config = model_config
        )

        resp = response.text
        return json.loads(resp)

    def get_context(i, model_output):

        lines = [f"Iteration #{i + 1}"]

        for field in config.structured_output_schema:
            lines.append(f"{field}:")
            lines.append(model_output.get(field))

        model_context = "\n".join(lines)

        return model_context


class User:
    def get_context(simulation_result):
        insample = simulation_result['is']
        train = simulation_result['train']
        checks = insample['checks']

        for check in checks:
            name = check['name']

            if (name == 'LOW_SHARPE'):
                sharpe_limit = check['limit']
            elif (name == 'LOW_FITNESS'):
                fitness_limit = check['limit']
            elif (name == 'LOW_TURNOVER'):
                turnover_lower_limit = check['limit']
            elif (name == 'HIGH_TURNOVER'):
                turnover_upper_limit = check['limit']
            elif (name == 'CONCENTRATED_WEIGHT'):
                weight_concentration = check

        train_sharpe = train['sharpe']
        train_fitness = train['fitness']
        train_turnover = round(100 * train['turnover'], 2)
        sub_universe_robustness = processing.sub_universe_robustness(simulation_result)
        alpha_quality_factor = round(processing.alpha_quality_factor(simulation_result, sharpe_limit, fitness_limit), 2)
        romad = round(insample['returns'] / insample['drawdown'], 2)
        turnover_stability = round(processing.turnover_stability(simulation_result), 2)

        user_contexts = [
                        "Simulation Results",
                        f"Train Period Sharpe: {train_sharpe}",
                        f"Train Period Fitness: {train_fitness}",
                        f"Train Period Turnover: {train_turnover}%",
                        f"Sub Universe Robustness: {sub_universe_robustness}",
                        f"Alpha Quality Factor: {alpha_quality_factor}",
                        f"RoMaD: {romad}",
                        f"Turnover Stability: {turnover_stability}"
                    ]

        if (train_sharpe < sharpe_limit):
            user_contexts[1] += f" is less than {sharpe_limit}"
        if (train_fitness < fitness_limit):
            user_contexts[2] += f" is less than {fitness_limit}"
        if (train['turnover'] < turnover_lower_limit):
            user_contexts[3] += f" is less than {round(100 * turnover_lower_limit, 2)}%"
        if (train['turnover'] > turnover_upper_limit):
            user_contexts[3] += f" is more than {round(100 * turnover_upper_limit, 2)}%"
        if (sub_universe_robustness is not None and sub_universe_robustness < 0.75):
            user_contexts[4] += f" is less than 0.75"
        if (alpha_quality_factor < 1):
            user_contexts[5] += f" is less than 1"
        if (romad < 2):
            user_contexts[6] += f" is less than 2"
        if (turnover_stability < 0.85):
            user_contexts[7] += f" is less than 0.85"

        if (train_sharpe >= sharpe_limit and
            train_fitness >= fitness_limit and
            train['turnover'] >= turnover_lower_limit and
            train['turnover'] <= turnover_upper_limit and
            alpha_quality_factor >= 1 and
            romad >= 2 and
            turnover_stability >= 0.85):
            if (sub_universe_robustness is not None and sub_universe_robustness >= 0.75):
                console.print("ITERATIONS SUCCESSFUL.", color = 'green')
                exit()

#         user_context = f"""
# Simulation Results:
# Train Period Sharpe: {train['sharpe']}
# Train Period Fitness: {train['fitness']}
# Train Period Turnover: {round(100 * train['turnover'], 2)}%
# Sub Universe Robustness: {processing.sub_universe_robustness(simulation_result)}
# Alpha Quality Factor: {round(processing.alpha_quality_factor(simulation_result), 2)}
# RoMaD: {round(insample['returns'] / insample['drawdown'], 2)}
# Turnover Stability: {round(processing.turnover_stability(simulation_result), 2)}
# """

        user_context = '\n'.join(user_contexts)

        if (weight_concentration['result'] == 'FAIL'):
            if weight_concentration.get('value'):
                user_context += f"\nWeight Concentration {round(weight_concentration['value'] * 100, 2)}% is above cutoff of {round(weight_concentration['limit'] * 100, 2)}%."
            else:
                user_context += "\nWeight is too strongly concentrated or too few instruments are assigned weight."

        if (train_sharpe < -1 * sharpe_limit / 2 or train_fitness < -1 * fitness_limit / 2):
            user_context += f"\nThe Hypothesis Direction is Reversed, Please Correct It."

        return user_context.strip()

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


console.print(initial_prompt, style = 'green')


for i in range(config.max_iterations):

    while True:

        try:
            model_output = Model.get_output(context)
            break

        except Exception as e:
            console.print(f"Model.get_output: {e}", style = 'red')
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

    user_context = User.get_context(simulation_result)

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