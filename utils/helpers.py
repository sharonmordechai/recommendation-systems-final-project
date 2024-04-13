def _repr_model(model):
    net_representation = '[uninitialised]' if model._net is None else repr(model._net)
    return f'<{model.__class__.__name__}: {net_representation}>'
