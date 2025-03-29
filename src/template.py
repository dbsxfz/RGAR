# This file is adapted from [Teddy-XiongGZ/MedRAG]  
# Original source: [https://github.com/Teddy-XiongGZ/MedRAG/blob/main/src/template.py]  
# we add prompts for no cot methods
from liquid import Template

general_cot_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_cot = Template('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json:
''')

general_medrag_system = '''You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Please first think step-by-step and then choose the answer from the provided options. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_medrag = Template('''
Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in one json:
''')

# General CoT System Template
general_cot_system2 = '''
You are a helpful medical expert, and your task is to answer a multi-choice medical question. Organize your output in a json formatted as Dict{"answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer. Please just give me the json of the answer
'''

# General CoT Template
general_cot2 = Template('''
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please just give me the json of the answer. Generate your output in json:
''')

# General MedRAG System Template
general_medrag_system2 = '''
You are a helpful medical expert, and your task is to answer a multi-choice medical question using the relevant documents. Organize your output in a json formatted as Dict{"answer_choice": Str{A/B/C/...}}. Your responses will be used for research purposes only, so please have a definite answer. Please just give me the json of the answer
'''

# General MedRAG Template
general_medrag2 = Template('''
Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please just give me the json of the answer. Generate your output in one json:
''')

general_extract = Template('''
Here are the relevant knowledge sources:
{{context}}

Here are the electronic health records:
{{ehr}}

Here is the question:
{{question}}

Please analyze and extract the key factual information in the electronic health records relevant to solving this question and present it as a Python list. 
Use concise descriptions for each item, formatted as ["key detail 1", ..., "key detail N"]. For example, ['Patient age: 39 years (Middle-aged)', 'Symptoms: fever, chills, left lower quadrant abdominal pain', 'Vital signs: high temperature (39.1°C or 102.3°F), tachycardia (pulse 126/min), tachypnea (respirations 28/min) and hypotension (blood pressure 80/50 mmHg)', 'Physical exam findings: mucopurulent discharge from the cervical os and left adnexal tenderness', 'Laboratory results: low platelet count (14,200/mm^3), elevated D-dimer (965 ng/mL)', 'Phenol test result: identification of a phosphorylated N-acetylglucosame dimmer with 6 fatty acids attached to a polysaccharide side chain'].Please only give me the list. Here is the list:
''')

general_extract_nolist = Template('''
Here are the relevant knowledge sources:
{{context}}

Here are the electronic health records:
{{ehr}}

Here is the question:
{{question}}

Please analyze and extract the key factual information in the electronic health records relevant to solving this question. Use concise descriptions for each item. For example, 'Patient age: 39 years (Middle-aged)', 'Symptoms: fever, chills, left lower quadrant abdominal pain', 'Vital signs: high temperature (39.1°C or 102.3°F), tachycardia (pulse 126/min), tachypnea (respirations 28/min) and hypotension (blood pressure 80/50 mmHg)', 'Physical exam findings: mucopurulent discharge from the cervical os and left adnexal tenderness', 'Laboratory results: low platelet count (14,200/mm^3), elevated D-dimer (965 ng/mL)', 'Phenol test result: identification of a phosphorylated N-acetylglucosame dimmer with 6 fatty acids attached to a polysaccharide side chain'. Please only give me the descriptions. Here are the descriptions:
''')



meditron_cot = Template('''
### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

meditron_medrag = Template('''
Here are the relevant documents:
{{context}}

### User:
Here is the question:
...

Here are the potential choices:
A. ...
B. ...
C. ...
D. ...
X. ...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Here are the potential choices:
{{options}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

simple_medrag_system = '''You are a helpful medical expert, and your task is to answer a medical question using the relevant documents.'''
simple_medrag_prompt = Template('''Here are the relevant documents:\n{{context}}\nHere is the question:\n{{question}}''')

i_medrag_system = '''You are a helpful medical assistant, and your task is to answer the given question following the instructions given by the user. '''

follow_up_instruction_ask = '''Please first analyze all the information in a section named Analysis (## Analysis). Then, use key terms from previous answers to form specific and direct questions. Generate {} concise, context-specific queries to search for additional information in an external knowledge base, in a section named Queries (## Queries). Each query should be simple and focused, directly relating to the key terms used in the answers. Wait for responses from the user before proceeding.'''
follow_up_instruction_answer = '''Please first think step-by-step to analyze all the information in a section named Analysis (## Analysis). Then, please provide your answer choice in a section named Answer (## Answer).'''