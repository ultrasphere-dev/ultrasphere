from sybil import Sybil
from sybil.parsers.myst import DocTestDirectiveParser as MarkdownDocTestParser
from sybil.parsers.myst import PythonCodeBlockParser as MarkdownPythonCodeBlockParser
from sybil.parsers.rest import DocTestParser as ReSTDocTestParser
from sybil.parsers.rest import PythonCodeBlockParser as ReSTPythonCodeBlockParser

markdown_examples = Sybil(
    parsers=[
        MarkdownDocTestParser(),
        MarkdownPythonCodeBlockParser(),
    ],
    patterns=["*.md"],
)

rest_examples = Sybil(
    parsers=[
        ReSTDocTestParser(),
        ReSTPythonCodeBlockParser(),
    ],
    patterns=["*.py", "*.rst"],
)

pytest_collect_file = (markdown_examples + rest_examples).pytest()
