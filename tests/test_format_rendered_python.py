"""Generated page formatting must preserve markup, interpolation and behavior."""
import ast

import pytest

from scripts.format_rendered_python import format_rendered_python


def test_display_markup_constants_and_idempotence():
    markup = '<details><summary>Help</summary>\n' + ('\\x & <b> exact  spaces </b> ' * 30) + '\n</details>'
    source = 'INTRO = ' + repr(markup) + '\nrender(INTRO)\n'
    result = format_rendered_python(source)
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(result))
    assert ast.literal_eval(ast.parse(result).body[0].value) == markup
    assert max(map(len, result.splitlines())) <= 180
    assert format_rendered_python(result) == result


def test_metadata_and_multiple_strings_on_one_line_fail_closed():
    source = 'metadata = ' + repr([{'title': 'Title', 'icon': str(i)} for i in range(20)]) + '\n'
    result = format_rendered_python(source)
    assert ast.dump(ast.parse(source)) == ast.dump(ast.parse(result))
    with pytest.raises(ValueError):
        format_rendered_python('x = ' + repr('a' * 300) + ' + ' + repr('b' * 300))


def test_formatted_strings_fail_closed():
    with pytest.raises((ValueError, SyntaxError)):
        format_rendered_python('x = f"' + 'a' * 300 + '{compute()}"\n')
