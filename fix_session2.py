keys = ['raw_data','clean_data','column_mapping','cleaning_config','cleaning_audit','data_dictionary','customer_features','rfm_results','lrfm_results','demographic_profile','segmentation_runs','selected_segmentation_model','segmented_customers','churn_dataset','churn_model','churn_model_metadata','churn_scores','retention_results','segment_history','global_filters']
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
s2 = s[:start] + new_block + s[end:]
open('utils/constants.py','w').write(s2)
print('done')
