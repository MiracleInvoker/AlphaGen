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
    'Alpha Expression': 'Write the Alpha Expression - a mathematical formula that MUST include descriptive variables and multi-line expressions. In a multi-line expression, every line must end with a semicolon (;).',
    'Hypothesis': 'Explain the Rationale behind the Idea - both from a Quantitative and Economic Perspective. Address both the quantitative justification and the economic intuition clearly.',
    'Implementation': 'Explain the use of the data fields and operators in the expression.',
    'Iteration Changes': 'If this is an iteration, list ALL changes explicitly and explain why each was introduced, referring to the simulation results that motivated the change.'
}


pnl_chart = {
    'train_color': '#5acdd5',
    'test_color': '#e57500',
    'drawdown_color': '#d32f2f'
}


initial_prompt = initial_prompt.format_map(simulation_settings)
initial_prompt = initial_prompt.strip()