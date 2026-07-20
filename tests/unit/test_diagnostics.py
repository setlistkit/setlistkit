"""Tests for the shared Diagnostic type and its renderer."""

import pytest

from setlistkit.diagnostics import (
    ERROR,
    WARNING,
    Diagnostic,
    DiagnosticError,
    render,
)


def test_render_code_frame_with_caret():
    diag = Diagnostic(
        severity=ERROR,
        summary="bad",
        path="f.json",
        line=2,
        col=2,
        length=2,
        caret_caption="empty",
        source="AAAA\nBBBB\nCCCC",
        detail="line one\nline two",
    )
    expected = "\n".join([
        "error: bad",
        "",
        "  f.json:2:2",
        "  1 | AAAA",
        "  2 | BBBB",
        "    |  ^^ empty",
        "  3 | CCCC",
        "",
        "  line one",
        "  line two",
    ])
    assert render(diag) == expected


def test_render_location_without_source_omits_frame():
    diag = Diagnostic(severity=ERROR, summary="boom", path="x.toml", line=7, col=3)
    out = render(diag)
    assert "  x.toml:7:3" in out
    # No source text was supplied, so there is no numbered code frame.
    assert " | " not in out


def test_render_headline_only():
    diag = Diagnostic(severity=WARNING, summary="just a warning")
    assert render(diag) == "warning: just a warning"


def test_render_path_without_line():
    diag = Diagnostic(severity=ERROR, summary="no config", path="/etc/x")
    out = render(diag)
    assert "  /etc/x" in out
    assert ":" not in out.split("/etc/x")[1]  # nothing appended after the bare path


def test_out_of_range_line_degrades_to_no_frame():
    diag = Diagnostic(severity=ERROR, summary="oops", path="f", line=99,
                      source="only one line")
    out = render(diag)
    assert " | " not in out


def test_unknown_severity_rejected():
    with pytest.raises(ValueError):
        Diagnostic(severity="catastrophe", summary="nope")


def test_is_error_flag():
    assert Diagnostic(ERROR, "x").is_error is True
    assert Diagnostic(WARNING, "x").is_error is False


def test_to_dict_omits_source_and_empty_fields():
    diag = Diagnostic(ERROR, "s", path="p", line=1, col=2, caret_caption="c",
                      detail="d", source="SRC")
    d = diag.to_dict()
    assert d == {
        "severity": "error", "summary": "s", "path": "p",
        "line": 1, "col": 2, "caret_caption": "c", "detail": "d",
    }
    assert "source" not in d


def test_equality_ignores_source():
    a = Diagnostic(ERROR, "same", path="p", line=1, source="one text")
    b = Diagnostic(ERROR, "same", path="p", line=1, source="different text")
    assert a == b


def test_diagnostic_error_carries_diagnostic():
    diag = Diagnostic(ERROR, "carried")
    err = DiagnosticError(diag)
    assert err.diagnostic is diag
    assert "carried" in str(err)
