s = open('services/dashboard_service.py').read()
old = "            moves.append(\n                {\n                    \"model\": model,\n                    \"period_from\": period_from,\n                    \"period_to\": period_to,\n                    \"segment_from\": record[\"segment_from\"],\n                    \"segment_to\": record[\"segment_to\"],\n                    \"customers\": int(record[\"customers\"]),\n                }\n            )"
new = "            moves.append(\n                {\n                    \"model\": model,\n                    \"span\": \"consecutive\",\n                    \"period_from\": period_from,\n                    \"period_to\": period_to,\n                    \"segment_from\": record[\"segment_from\"],\n                    \"segment_to\": record[\"segment_to\"],\n                    \"customers\": int(record[\"customers\"]),\n                }\n            )"
if old not in s:
    print('no')
else:
    s2 = s.replace(old, new)
    open('services/dashboard_service.py','w').write(s2)
    print('ok')
