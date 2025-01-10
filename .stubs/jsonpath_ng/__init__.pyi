from typing import Any

class Fields:
    fields: tuple[str, ...]

class DatumInContext:
    value: Any
    fields: Fields
    context: "DatumInContext"

class Jsonpath:
    def find(self, /, value: Any) -> list[DatumInContext]: ...

def parse(jsonpath: str, /) -> Jsonpath: ...
