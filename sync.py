keys = ['raw_data','clean_data','column_mapping','cleaning_config','cleaning_audit','data_dictionary','customer_features','rfm_results','lrfm_results','demographic_profile','segmentation_runs','selected_segmentation_model','segmented_customers','churn_dataset','churn_model','churn_model_metadata','churn_scores','retention_results','segment_history','global_filters']
# constants
s = open('utils/constants.py').read()
start = s.find('SESSION_KEYS')
eq = s.find('=', start)
op = s.find('(', eq)
depth = 0
i = op
end = op
while i < len(s):
    if s[i] == '(':
        depth += 1
    elif s[i] == ')':
        depth -= 1
        if depth == 0:
            end = i + 1
            break
    i += 1
new_block = 'SESSION_KEYS = (\n    "' + '",\n    "'.join(keys) + '",\n)'
open('utils/constants.py','w').write(s[:start] + new_block + s[end:])
# state
s = open('services/state_service.py').read()
start = s.find('_DEFAULTS')
eq = s.find('=', start)
op = s.find('{', eq)
depth = 0
i = op
end = op
while i < len(s):
    if s[i] == '{':
        depth += 1
    elif s[i] == '}':
        depth -= 1
        if depth == 0:
            end = i + 1
            break
    i += 1
defs = []
for k in keys:
    if k in ['raw_data','clean_data','customer_features','rfm_results','lrfm_results','selected_segmentation_model','segmented_customers','churn_model','churn_scores']:
        defs.append(f'    "{k}": None')
    elif k in ['segmentation_runs']:
        defs.append(f'    "{k}": []')
    else:
        defs.append(f'    "{k}": {{}}')
newd = '{\n' + ',\n'.join(defs) + ',\n}'
open('services/state_service.py','w').write(s[:start] + newd + s[end:])
print('ok')
