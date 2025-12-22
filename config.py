simulation_settings = {
    "instrumentType": "EQUITY",
    "region": "USA",
    "universe": "TOP3000",
    "delay": 1,
    "decay": 0,
    "neutralization": "SUBINDUSTRY",
    "truncation": 0.08,
    "pasteurization": "ON",
    "testPeriod": "P2Y",
    "unitHandling": "VERIFY",
    "nanHandling": "ON",
    "language": "FASTEXPR",
    "visualization": False,
    "maxTrade": "OFF",
}


data_fields = ["liabilities", "assets"]

initial_prompt = """
Simulation Settings:
Region: {region}
Universe: {universe}
Delay: {delay}
Neutralization: {neutralization}
Max Trade: {maxTrade}

you MUST start your Iteration #1 with:
liabilities / assets
"""


model = "gemini-3-flash-preview"
temperature = 1
thinking_level = "high"
system_prompt_file = "system_prompts/alpha.txt"

operators_file = "operators.txt"
max_iterations = 200


structured_output_schema_description = (
    "Schema for Structured Output of Alpha Expression."
)
structured_output_schema = {
    "Alpha Expression": "Output the Alpha Expression that MUST include descriptive variables and multi-line expressions. In a multi-line expression, every line must end with a semicolon (;).",
    "Hypothesis": "Explain the Rationale behind the Idea - both from a Quantitative and Economic Perspective. Address both the quantitative justification and the economic intuition clearly.",
    "Implementation": "Explain the use of the data fields and operators in the expression.",
    "Iteration Changes": "If this is an iteration, list ALL changes explicitly and explain why each was introduced, referring to the simulation results that motivated the change.",
}


pnl_chart = {
    "file_name": "pnl_chart.png",
    "test": 252 * 2,
    "train_color": "#5acdd5",
    "test_color": "#e57500",
    "secondary_color": "#60ca68",
    "tertiary_color": "#acd147",
}


if simulation_settings["region"] == "ASI":
    simulation_settings["maxTrade"] = "ON"


initial_prompt = initial_prompt.format_map(simulation_settings)
initial_prompt = initial_prompt.strip()
