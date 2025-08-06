simulation_settings = {
    'instrumentType': 'EQUITY',
    'region': 'USA',
    'universe': 'TOP3000',
    'delay': 1,
    'decay': 0,
    'neutralization': 'SECTOR',
    'truncation': 0,
    'pasteurization': 'ON',
    'testPeriod': 'P1Y',
    'unitHandling': 'VERIFY',
    'nanHandling': 'ON',
    'language': 'FASTEXPR',
    'visualization': False,
}


data_fields = ['fnd6_acodo', 'assets']

initial_prompt = '''
Simulation Settings:
Region: {region}
Universe: {universe}
Neutralization: {neutralization}
'''

model = 'gemini-2.5-pro'
temperature = 1
thinking_budget = 32768
system_prompt_file = 'prompts/temp.txt'


operators_file = 'operators.txt'
max_iterations = 100


structured_output_schema_description = 'Schema for Structured Output of Alpha Expression.'
structured_output_schema = {
    'Alpha Expression': 'Alphas are Mathematical models that seek to predict the future price movement of various financial instruments.',
    'Hypothesis': 'Explain the Rationale behind the Idea - both from a Quantitative and Economic Perspective.',
    'Implementation': 'Explain the Use of the Data Fields and Operators.',
    'Iteration Changes': 'If it is an Iteration, highlight ALL the Changes and Explain why they were introduced in light of Simulation Results.'
}


pnl_chart = {
    'train_color': '#5acdd5',
    'test_color': '#e57500',
    'drawdown_color': '#d32f2f'
}


initial_prompt = initial_prompt.format_map(simulation_settings)
initial_prompt = initial_prompt.strip()