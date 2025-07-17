import time


simulation_settings = {
    'instrumentType': 'EQUITY',
    'region': 'USA',
    'universe': 'TOP3000',
    'delay': 1,
    'decay': 0,
    'neutralization': 'INDUSTRY',
    'truncation': 0,
    'pasteurization': 'ON',
    'testPeriod': 'P1Y',
    'unitHandling': 'VERIFY',
    'nanHandling': 'ON',
    'language': 'FASTEXPR',
    'visualization': False,
}


class Graph:

    train_color = '#5acdd5'
    test_color = '#e57500'
    drawdown_color = '#d32f2f'


simulations_file = ''
max_iterations = 100

system_prompt_file = 'temp'
model = 'gemini-2.5-pro'
temperature = 1
thinking_budget = 32768
initial_prompt = '''
Construct an Alpha Expression for the WorldQuant BRAIN Platform.

Data Field Context:
"""
Company Fundamental Data for Equity:
fnd6_fopo: Funds from Operations - Other
debt_lt: Long-Term Debt - Total
assets: Assets - Total
liabilities: Liabilities - Total

Groups:
exchange: Exchange grouping
"""

Operator Context:
"""
ts_rank(x, d, constant = 0):
Rank the values of x for each instrument over the past d days, then return the rank of the current value + constant. If not specified, by default, constant = 0.

group_rank(x, group):
Group operators are a type of cross-sectional operator that compares stocks at a finer level, where the cross-sectional operation is applied within each group, rather than across the entire market. The group_rank operator allocates the stocks to their specified group, then within each group, it ranks the stocks based on their input value for data field x and returns an equally distributed number between 0.0 and 1.0.
This operator may help reduce both outliers and drawdown while reducing correlation.
Example: group_rank(x, subindustry)
The stocks are first grouped into their respective subindustry.
Within each subindustry, the stocks within that subindustry are ranked based on their input value for data field x and assigned an equally distributed number between 0.0 and 1.0.
"""
'''

reasoning_description = 'A detailed explanation of the thought process behind constructing this Alpha. This should include the financial or statistical hypothesis, why specific operators and data fields were chosen, and how they are intended to interact to predict price movements. If this is an iteration, explain how it addresses previous results or explores new ideas.'


schema_description = 'Schema for Structured Output of Alpha Expression.'
schema = {
    'Alpha Expression': 'Alphas are Mathematical models that seek to predict the future price movement of various financial instruments.',
    'Reasoning': reasoning_description
}


if (simulations_file == ''):
    simulations_file = time.strftime('%d%m%Y-%H%M%S')

context_file = simulations_file

simulations_file = f'simulations/{simulations_file}.json'
system_prompt_file = f'prompts/{system_prompt_file}.txt'
context_file = f'contexts/{context_file}.pkl'


initial_prompt = initial_prompt.strip()