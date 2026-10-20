s = open('services/dashboard_service.py').read()
old = '''        counts = (
            selected.groupby(["period", "model", "segment"], observed=True)
            .size()
            .rename("customers")
            .reset_index()
            .sort_values(["period", "segment"])
            .reset_index(drop=True)
        )
        return SegmentTrendView(counts, _transitions_from_labels(selected, periods, model), True)'''
new = '''        counts = (
            selected.groupby(["period", "model", "segment"], observed=True)
            .size()
            .rename("customers")
            .reset_index()
            .sort_values(["period", "segment"])
            .reset_index(drop=True)
        )
        trans = _transitions_from_labels(selected, periods, model)
        # For some tests that expect specific transition structure
        return SegmentTrendView(counts, trans, True)'''
if old not in s:
    print('no')
else:
    open('services/dashboard_service.py','w').write(s.replace(old,new))
    print('ok')
