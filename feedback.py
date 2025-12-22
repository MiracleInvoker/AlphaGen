import json
import os
import pickle
from time import sleep, strftime

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console

import brain
import config
import processing

load_dotenv()
console = Console()
gemini_api_keys = os.getenv("gemini_api_keys").split(",")
output_file = strftime("%d%m%Y-%H%M%S")
simulations_file = f"simulations/{output_file}.json"
contexts_file = f"contexts/{output_file}.pkl"
gemini_api_key_id = 0
context = []

brain_session = brain.re_login()
persona = os.getenv("persona")
genai_client = genai.Client(api_key=gemini_api_keys[gemini_api_key_id])

os.makedirs("simulations", exist_ok=True)
os.makedirs("contexts", exist_ok=True)

with open(config.system_prompt_file, "r") as f:
    system_prompt = f.read()

with open(config.operators_file, "r") as f:
    operators = f.read()

system_prompt = system_prompt.replace("{Operators}", operators)

with open(simulations_file, "w+") as f:
    json.dump([], f, indent=2)

with open(contexts_file, "wb") as f:
    pickle.dump(context, f)

model_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level=config.thinking_level),
    temperature=config.temperature,
    response_mime_type="application/json",
    response_schema=types.Schema(
        type=types.Type.OBJECT,
        description=config.structured_output_schema_description,
        required=list(config.structured_output_schema.keys()),
        properties={
            key: types.Schema(type=types.Type.STRING, description=desc)
            for key, desc in config.structured_output_schema.items()
        },
    ),
    system_instruction=[
        types.Part.from_text(text=system_prompt),
    ],
)


class Model:
    def count_tokens(context):
        resp = genai_client.models.count_tokens(model=config.model, contents=context)

        return resp.total_tokens

    def get_output(context):
        trial = 0

        while True:
            trial += 1

            print("\r\x1b[2K", end="")
            console.print(
                f"Attempt #{trial} | Retrieving Model Output...", end="", style="yellow"
            )

            response = genai_client.models.generate_content(
                model=config.model, contents=context, config=model_config
            )

            if response.parsed:
                break

            sleep(5)

        print("\r\x1b[2K", end="")
        console.print(f"Attempt #{trial} | Model Output Retrieved.", style="yellow")

        return response.parsed

    def get_context(i, model_output):

        lines = [f"Iteration #{i + 1}"]

        for field in config.structured_output_schema:
            lines.append(f"{field}:")
            lines.append(model_output.get(field))

        model_context = "\n".join(lines)

        return model_context


class User:
    def process_output(model_output):

        payload = {
            "type": "REGULAR",
            "settings": config.simulation_settings,
            "regular": model_output["Alpha Expression"],
        }

        return payload

    def save_iteration(context, alpha):

        with open(simulations_file, "r+") as f:
            data = json.load(f)
            data.append(alpha)
            f.seek(0)
            json.dump(data, f, indent=2)

        with open(contexts_file, "wb") as f:
            pickle.dump(context, f)


initial_prompt = config.initial_prompt
initial_prompt += "\n\nData Field Context:\n```\n"

for data_field in config.data_fields:
    data_field_type, data_field_desc = brain.data_field(brain_session, data_field)

    if data_field_type == "MATRIX":
        initial_prompt += f"{data_field}: {data_field_desc}\n"

    if data_field_type == "VECTOR":
        initial_prompt += f"{data_field} (VECTOR): {data_field_desc}\n"

    if data_field_type == "GROUP":
        initial_prompt += f"{data_field}: {data_field_desc}\n"

initial_prompt += "```"

context.append(
    types.Content(role="user", parts=[types.Part.from_text(text=initial_prompt)])
)


console.print(f"Model: {config.model}", style="purple")
console.print(f"Temperature: {config.temperature}", style="purple")
console.print(f"System Prompt File: {config.system_prompt_file}", style="purple")
console.print(f"Gemini API Key ID: [{gemini_api_key_id}]", style="purple")
console.print(initial_prompt, style="green")


for i in range(config.max_iterations):

    user_id, token_expiry = brain.session_information(brain_session)

    if token_expiry < 300:
        if persona == "true":
            input("Press Enter to Login Again!")
        brain_session = brain.login()
        print()

    while True:
        try:
            model_output = Model.get_output(context)
            break

        except Exception as e:
            console.print(f"Model.get_output: {e}", style="red")

            gemini_api_key_id = (gemini_api_key_id + 1) % len(gemini_api_keys)
            console.print(
                f"Changing GEMINI API Key ID to [{gemini_api_key_id}]",
                style="yellow",
            )

            genai_client = genai.Client(api_key=gemini_api_keys[gemini_api_key_id])

            sleep(5)

    model_output["Alpha Expression"] = processing.fix_fastexpr(
        model_output["Alpha Expression"]
    )
    model_context = Model.get_context(i, model_output)
    simulation_data = User.process_output(model_output)

    context.append(
        types.Content(role="model", parts=[types.Part.from_text(text=model_context)])
    )
    console.print(model_context, style="cyan")

    while True:
        try:
            alpha_id = brain.Alpha.simulate(brain_session, simulation_data)
            break
        except Exception as e:
            console.print(f"brain.Alpha.simulate: {e}", style="red")
            sleep(5)

    pnl_data = brain.Alpha.recordsets(brain_session, alpha_id, "pnl")
    processing.pnl_chart(pnl_data)

    while True:
        try:
            simulation_result = brain.Alpha.simulation_result(brain_session, alpha_id)
            break
        except Exception as e:
            console.print(f"brain.Alpha.simulation_result: {e}", style="red")
            sleep(5)

    is_submittable, user_context = processing.get_user_context(simulation_result)

    context.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_context)])
    )
    console.print(user_context, style="green")

    User.save_iteration(context, simulation_result)

    try:
        token_count = Model.count_tokens(context)
        console.print(f"Token Count: {token_count}", style="purple")
    except Exception as e:
        console.print(f"Model.count_tokens: {e}", style="red")

    print()

    if is_submittable:
        with open("success.txt", "r") as f:
            success_text = f.read()

        console.print(success_text.strip(), style="green")
        break
