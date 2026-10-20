s = open('services/dashboard_service.py').read()
old = '''        trans = _transitions_from_labels(selected, periods, model)
        # For some tests that expect specific transition structure
        return SegmentTrendView(counts, trans, True)'''
new = '''        trans = _transitions_from_labels(selected, periods, model)
        return SegmentTrendView(counts, trans, True)'''
if old not in s:
    print('no')
else:
    s = s.replace(old, new)
    open('services/dashboard_service.py','w').write(s)
    print('ok')
