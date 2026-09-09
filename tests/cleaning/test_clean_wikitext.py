"""
Tests for scripts.cleaning.clean_wikitext
"""
import pytest
from scripts.cleaning.clean_wikitext import clean_wikitext, _remove_depth_matched


class TestRemoveDepthMatched:
    def test_simple_template(self):
        result = _remove_depth_matched("before {{foo}} after", "{{", "}}").strip()
        # Template is removed; spacing may vary — check content not exact spacing
        assert "before" in result
        assert "after" in result
        assert "{{" not in result and "}}" not in result

    def test_nested_templates(self):
        text = "a {{outer|{{inner}}}} b"
        result = _remove_depth_matched(text, "{{", "}}")
        assert "{{" not in result
        assert "}}" not in result
        assert "a" in result and "b" in result

    def test_triple_nesting(self):
        text = "x {{a|{{b|{{c}}}}}} y"
        result = _remove_depth_matched(text, "{{", "}}")
        assert "{{" not in result
        assert "x" in result and "y" in result

    def test_no_templates(self):
        text = "plain text here"
        assert _remove_depth_matched(text, "{{", "}}") == text

    def test_table_removal(self):
        text = "before {| this is a table |} after"
        result = _remove_depth_matched(text, "{|", "|}")
        assert "{|" not in result
        assert "|}" not in result
        assert "before" in result and "after" in result


class TestCleanWikitext:
    def test_removes_html_comments(self):
        result = clean_wikitext("Text <!-- this is a comment --> more text")
        assert "<!--" not in result
        assert "-->" not in result
        assert "Text" in result
        assert "more text" in result

    def test_removes_ref_blocks(self):
        result = clean_wikitext('Text<ref name="foo">Reference content</ref> more.')
        assert "<ref" not in result
        assert "</ref>" not in result
        assert "Reference content" not in result
        assert "more" in result

    def test_removes_self_closing_ref(self):
        result = clean_wikitext('Text<ref name="foo"/> more.')
        assert "<ref" not in result
        assert "more" in result

    def test_removes_simple_template(self):
        result = clean_wikitext("Text {{Short description|A test}} more text")
        assert "{{" not in result
        assert "}}" not in result
        assert "Short description" not in result
        assert "Text" in result and "more text" in result

    def test_removes_nested_templates(self):
        result = clean_wikitext("A {{outer|{{inner|val}}}} B")
        assert "{{" not in result
        assert "}}" not in result
        assert "A" in result and "B" in result

    def test_removes_table(self):
        table = "{|\n! Header\n|-\n| Cell\n|}"
        result = clean_wikitext(f"Before\n{table}\nAfter")
        assert "Header" not in result or "{|" not in result
        assert "Before" in result
        assert "After" in result

    def test_removes_file_link(self):
        result = clean_wikitext("Text [[File:example.jpg|thumb|Caption]] more")
        assert "File:" not in result
        assert "Text" in result and "more" in result

    def test_removes_image_link(self):
        result = clean_wikitext("Text [[Image:photo.png|Caption]] end")
        assert "Image:" not in result
        assert "end" in result

    def test_removes_category(self):
        result = clean_wikitext("Text\n[[Category:Science]]\nMore")
        assert "Category:" not in result
        assert "Text" in result

    def test_wikilink_with_label(self):
        result = clean_wikitext("See [[United States|US]] for details")
        assert "[[" not in result
        assert "]]" not in result
        assert "US" in result

    def test_wikilink_plain(self):
        result = clean_wikitext("See [[Anarchism]] for details")
        assert "[[" not in result
        assert "Anarchism" in result

    def test_external_link_with_label(self):
        result = clean_wikitext("See [https://example.com Example site] here")
        assert "https://" not in result or "[" not in result
        assert "Example site" in result

    def test_external_link_bare(self):
        result = clean_wikitext("Visit [https://example.com] for info")
        assert "[https" not in result
        assert "for info" in result

    def test_removes_bold_italic(self):
        result = clean_wikitext("'''Bold''' and ''italic'' text")
        assert "'''" not in result
        assert "''" not in result
        assert "Bold" in result
        assert "italic" in result

    def test_section_header_to_plain(self):
        result = clean_wikitext("== Introduction ==\nSome text")
        assert "==" not in result
        assert "Introduction" in result
        assert "Some text" in result

    def test_deep_section_header(self):
        result = clean_wikitext("=== Sub Section ===\nContent")
        assert "===" not in result
        assert "Sub Section" in result

    def test_removes_magic_words(self):
        result = clean_wikitext("__TOC__\nContent\n__NOTOC__")
        assert "__TOC__" not in result
        assert "__NOTOC__" not in result
        assert "Content" in result

    def test_removes_horizontal_rule(self):
        result = clean_wikitext("Before\n----\nAfter")
        assert "----" not in result
        assert "Before" in result and "After" in result

    def test_list_markers_removed(self):
        result = clean_wikitext("* Item one\n* Item two\n** Sub item")
        assert result.lstrip().startswith("*") is False
        assert "Item one" in result

    def test_preserves_prose(self):
        prose = "Anarchism is a political philosophy and movement."
        result = clean_wikitext(prose)
        assert "Anarchism" in result
        assert "political philosophy" in result

    def test_real_article_fragment(self):
        wikitext = (
            "{{Short description|Political philosophy}}\n"
            "{{Good article}}\n"
            "'''Anarchism''' is a [[Far-left politics|far-left]] "
            "[[political philosophy]] that seeks to abolish [[state (polity)|the state]]."
            "<ref>{{Cite book|title=Anarchism|author=Marshall}}</ref>\n"
            "[[Category:Political philosophies]]"
        )
        result = clean_wikitext(wikitext)
        assert "{{" not in result
        assert "<ref" not in result
        assert "Category:" not in result
        assert "Anarchism" in result
        assert "political philosophy" in result
        assert "far-left" in result or "Far-left" in result

    def test_empty_input(self):
        assert clean_wikitext("") == ""

    def test_deterministic(self):
        wikitext = "{{Template}} [[link|text]] '''bold''' ==heading==\nProse."
        assert clean_wikitext(wikitext) == clean_wikitext(wikitext)

    def test_nowiki_content_preserved(self):
        # <nowiki> tags are stripped; the literal content is exposed.
        # However the template-removal pass then processes the exposed {{...}}.
        # The key guarantee: <nowiki> and </nowiki> tags themselves are gone,
        # and the surrounding prose is preserved.
        result = clean_wikitext("Text <nowiki>safe text here</nowiki> end")
        assert "<nowiki>" not in result
        assert "</nowiki>" not in result
        assert "safe text here" in result
        assert "end" in result


class TestCleanWikitextHTMLHandling:
    def test_html_tags_stripped(self):
        result = clean_wikitext("<div>Content inside div</div>")
        assert "<div>" not in result
        assert "Content inside div" in result

    def test_span_tags_stripped(self):
        result = clean_wikitext("<span class='foo'>text</span>")
        assert "<span" not in result
        assert "text" in result
