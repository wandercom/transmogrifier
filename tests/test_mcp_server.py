"""Check the real MCP entry point without starting its transport."""

from unittest.mock import Mock

import pytest

from transmogrifier import core, mcp_server

fastmcp = pytest.importorskip("mcp.server.fastmcp", reason="requires the supported mcp extra")


def test_main_registers_callable_tools_and_stdio_transport(monkeypatch):
    servers = []

    class Server:
        def __init__(self, name, **kwargs):
            self.name = name
            self.tools = {}
            self.transport = None
            servers.append(self)

        def tool(self):
            def register(function):
                self.tools[function.__name__] = function
                return function
            return register

        def run(self, *, transport):
            self.transport = transport

    translator = Mock()
    translator.translate.return_value.model_dump.return_value = {"output_text": "hello"}
    translator._detector.detect.return_value = (core.Register.direct, 1.0)
    translator._profile_cache.list_profiles.return_value = []
    monkeypatch.setattr(fastmcp, "FastMCP", Server)
    monkeypatch.setattr(core, "Transmogrifier", lambda: translator)

    mcp_server.main()
    server, = servers
    assert server.name == "transmogrifier"
    assert server.transport == "stdio"
    assert set(server.tools) == {"transmog_translate", "transmog_detect", "transmog_profiles"}
    assert server.tools["transmog_translate"]("hello", target_register="direct") == {"output_text": "hello"}
    assert translator.translate.call_args.kwargs["config"].target_register is core.Register.direct
    assert server.tools["transmog_detect"]("hello") == {"register": "direct", "confidence": 1.0}
    assert server.tools["transmog_profiles"]() == []
