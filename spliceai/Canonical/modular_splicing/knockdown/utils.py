def render_cell_line(line):
    if isinstance(line, str):
        return line
    raise RuntimeError(f"Unknown cell line: {line}")


def render_cell_lines(lines):
    lines = [render_cell_line(x) for x in lines]
    return ", ".join(lines)
