from typing import Any, Callable


class Router:
    def __init__(self):
        self._handlers: dict[str, Callable] = {}
    

    def on(self, command_name: str) -> Callable:
        def decorator(func: Callable) -> Callable:
            if command_name in self._handlers:
                raise ValueError(f'Handler for "{command_name}" is already registered.')

            self._handlers[command_name] = func

            return func
        
        return decorator
    

    async def dispatch(self, command_name: str, sid: str, ws, data: Any) -> None:
        handler = self._handlers.get(command_name)
        if not handler:
            print(
                f'Warning: unhandled command "{command_name}". '
                'If this is a valid command, ensure it\'s mapped to a function'
                )
            return
        
        await handler(sid, ws, data)