import json
import os
from time import sleep

from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console

import ace_lib as ace
import alpha_utils

load_dotenv()
console = Console()
context = []

gemini_api_keys = os.getenv("gemini_api_keys").split(",")
gemini_api_key_id = 0
genai_client = genai.Client(api_key=gemini_api_keys[gemini_api_key_id])

brain_session = alpha_utils.get_stored_session()


with open("config.json", "r") as f:
    config = json.load(f)

with open("alpha.json", "r") as f:
    alpha = json.load(f)

model_params = config["model_config"]
structured_output = config["structured_output"]


initial_prompt = "Data Field Context:"

for data_field in config["data_fields"]:
    data_field_df = ace.get_datafields(brain_session, search=data_field)

    item = data_field_df.iloc[0]

    d_type = item["type"]
    d_desc = item["description"]

    if d_type == "VECTOR" and "VECTOR" not in config["operators"]:
        config["operators"].append("VECTOR")

    initial_prompt += f"\n{data_field} ({d_type}): {d_desc}"

if "Group" in config["operators"]:
    groups_df = ace.get_datafields(
        brain_session,
        instrument_type="EQUITY",
        region=alpha["settings"]["region"],
        delay=alpha["settings"]["delay"],
        universe=alpha["settings"]["universe"],
        data_type="GROUP",
    )
    groups_df = groups_df.sort_values(by="alphaCount", ascending=False).head(6)
    initial_prompt += "\n"
    initial_prompt += "\n".join(f"{x} (GROUP)" for x in groups_df["id"])

context.append(
    types.Content(role="user", parts=[types.Part.from_text(text=initial_prompt)])
)


operators_str = "**Operators Context**\n\n"

for operator_category in config["operators"]:
    if operator_category[0] == "!":
        continue

    operators_str += f"# {operator_category}\n"

    with open(f"operators/{operator_category}.txt", "r") as f:
        operators_str += f.read()

    operators_str += "\n\n"

with open(model_params["system_prompt"], "r") as f:
    system_prompt = f.read()

system_prompt = system_prompt.replace("{operators}", operators_str.strip())


model_config = types.GenerateContentConfig(
    thinking_config=types.ThinkingConfig(thinking_level=model_params["thinking_level"]),
    temperature=model_params["temperature"],
    response_mime_type="application/json",
    response_schema=types.Schema(
        type=types.Type.OBJECT,
        description=structured_output["schema_description"],
        required=list(structured_output["schema"].keys()),
        properties={
            key: types.Schema(type=types.Type.STRING, description=desc)
            for key, desc in structured_output["schema"].items()
        },
    ),
    system_instruction=[types.Part.from_text(text=system_prompt)],
)


def get_model_output(context):
    trial = 0

    while True:
        trial += 1

        print("\r\x1b[2K", end="")
        console.print(
            f"Attempt #{trial} | Retrieving Model Output...", end="", style="yellow"
        )

        response = genai_client.models.generate_content(
            model=model_params["model"], contents=context, config=model_config
        )

        if response.parsed:
            break

        sleep(5)

    print("\r\x1b[2K", end="")
    console.print(f"Attempt #{trial} | Model Output Retrieved.", style="yellow")

    return response.parsed


def get_model_context(i, model_output):
    lines = [f"Iteration #{i + 1}"]
    for key, value in model_output.items():
        lines.append(f"{key}:")
        lines.append(str(value))
    return "\n".join(lines)


console.print(f"Model: {model_params["model"]}", style="purple")
console.print(f"Temperature: {model_params["temperature"]}", style="purple")
console.print(f"System Prompt File: {model_params["system_prompt"]}", style="purple")
console.print(f"Gemini API Key ID: [{gemini_api_key_id}]", style="purple")
console.print(initial_prompt, style="green")


for i in range(config["iterations"]):

    while True:
        try:
            model_output = get_model_output(context)
            break

        except Exception as e:
            console.print(f"\nget_model_output: {e}", style="red")

            gemini_api_key_id = (gemini_api_key_id + 1) % len(gemini_api_keys)
            console.print(
                f"Changing GEMINI API Key ID to [{gemini_api_key_id}]",
                style="yellow",
            )

            genai_client = genai.Client(api_key=gemini_api_keys[gemini_api_key_id])

            sleep(5)

    model_output["Alpha Expression"] = alpha_utils.fix_fastexpr(
        model_output["Alpha Expression"]
    )

    console.print(get_model_context(i, model_output), style="cyan")

    model_output.pop("Constraint Checklist")

    context.append(
        types.Content(
            role="model",
            parts=[types.Part.from_text(text=get_model_context(i, model_output))],
        )
    )

    alpha["regular"] = model_output["Alpha Expression"]

    simulation_results = ace.simulate_alpha_list(
        brain_session, [alpha], limit_of_concurrent_simulations=1
    )
    simulation_result = simulation_results[0]

    alpha_id = simulation_result["alpha_id"]

    pnl_df = ace.get_alpha_pnl(brain_session, alpha_id)
    alpha_utils.pnl_chart(config["pnl_chart"], pnl_df)

    full_simulation_result = ace.get_simulation_result_json(brain_session, alpha_id)

    is_submittable, user_context = alpha_utils.get_user_context(full_simulation_result)

    context.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_context)])
    )

    console.print(user_context, style="green")

    print()
    print()

    if is_submittable:
        console.print("Alpha Generated Successfully!", style="green")
        break
