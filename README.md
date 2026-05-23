# moduna-otel

Python OpenTelemetry helpers for Moduna AI traces.

```python
from moduna_otel import ModunaOTEL

otel = ModunaOTEL(agent_name="support-agent", framework="langchain")
handler = otel.langchain_handler({"conversation_id": "conversation-1"})
```
