from sybil import Sybil
from sybil.evaluators.doctest import NUMBER
from sybil.parsers.myst import DocTestDirectiveParser as MarkdownDocTestParser
from sybil.parsers.myst import PythonCodeBlockParser as MarkdownPythonCodeBlockParser
from sybil.parsers.myst import SkipParser as MarkdownSkipParser
from sybil.parsers.rest import DocTestParser as ReSTDocTestParser
from sybil.parsers.rest import PythonCodeBlockParser as ReSTPythonCodeBlockParser
from sybil.parsers.rest import SkipParser as ReSTSkipParser

markdown_examples = Sybil(
    parsers=[
        MarkdownDocTestParser(NUMBER),
        MarkdownPythonCodeBlockParser(doctest_optionflags=NUMBER),
        MarkdownSkipParser(),
    ],
    patterns=["*.md"],
)

rest_examples = Sybil(
    parsers=[
        ReSTDocTestParser(NUMBER),
        ReSTPythonCodeBlockParser(),
        ReSTSkipParser(),
    ],
    patterns=["*.py", "*.rst"],
)

pytest_collect_file = (markdown_examples + rest_examples).pytest()
