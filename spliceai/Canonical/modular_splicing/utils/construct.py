def construct(type_map, spec, **default_kwargs):
    spec = dict(spec.items())
    type_name = spec.pop("type")
    default_kwargs.update(spec)
    return type_map[type_name](**default_kwargs)
