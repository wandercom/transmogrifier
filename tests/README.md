# Test qualification

Run from the repository root with a dedicated virtual environment:

```bash
python -m pip install -e '.[dev,mcp,gemini]'
PYTHONPATH=src python -m pytest --import-mode=importlib -q \
  tests/test_*.py tests/smoke/test_transmogrifier_*.py
```

77 tests passed in the 2026-09-13 Python 3.13 qualification, with MCP 1.30.0
and google-genai 2.23.0 installed. This includes credential-free tests of the
actual MCP entry point and the Gemini backend against its declared SDK. The
optional integration tests skip explicitly if their extras are absent.

`tests/src_transmogrifier_*/contract_test.py` retain generated expectations for
source components. Imports and patch targets now point to the real package;
the backend factory suite no longer overwrites the entire package in
`sys.modules`. Many remaining tests still require obsolete constructor fields,
mock targets, or behaviors; some simulate an implementation inside the test
instead of invoking source. They are not independent release evidence merely
because an individual case passes.

`tests/contracts_*/` and `tests/smoke/test_contracts_*` target generated
interface scaffolds rather than the wheel's `transmogrifier` package. Several
scaffolds contain non-executable Python (`primitive`, undefined Enum,
Callable, Register, and ellipsis bodies). Recursive contracts-of-contracts
were added with Pact scaffolding in commit `4aa3e2a`.

Full-tree pytest remains **not green**. All draft tests and assertions remain
in place. No global skip, xfail, or collection exclusion hides these failures.
Porting their intended behavior remains separate work; fabricating the old
interfaces would not qualify the current package.

To inspect every retained artifact, run:

```bash
PYTHONPATH=src python -m pytest --import-mode=importlib -q tests
```
