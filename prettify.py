from IPython.display import Markdown, display


def pretty(text):
    display(Markdown(text))


def pretty_code(text):
    display(Markdown(f"```python\n{text}\n```"))


def pretty_code_block(text):
    display(Markdown(f"```python\n{text}\n```"))


def pretty_code_block_with_title(text, title):
    display(Markdown(f"```python\n{text}\n```"))
    display(Markdown(f"### {title}"))


def pretty_code_block_with_title_and_code(text, title):
    display(Markdown(f"```python\n{text}\n```"))
    display(Markdown(f"### {title}"))


def pretty_json(text):
    display(Markdown(f"```json\n{text}\n```"))
