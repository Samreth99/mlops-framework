import re
from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

_CSS = """
<style>
  .information-container .info .title small.version-stamp {
    display: none !important;
  }
  .information-container .info a[href$="openapi.json"] {
    display: none !important;
  }
</style>
"""

def mount_custom_docs(app: FastAPI, path: str = "/docs") -> None:
    @app.get(path, include_in_schema=False)
    async def custom_swagger_ui() -> HTMLResponse:  # type: ignore[func-returns-value]
        original = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title,
            # optional: avoid default favicon URL being set
            swagger_favicon_url="",
        )

        html = original.body.decode("utf-8")

        # 🔹 remove any favicon link tags
        html = re.sub(
            r'<link[^>]+rel="(?:shortcut )?icon"[^>]*>',
            "",
            html,
            flags=re.IGNORECASE,
        )

        # 🔹 inject our CSS
        html = html.replace("</head>", f"{_CSS}</head>")

        return HTMLResponse(
            content=html,
            status_code=original.status_code,
            media_type=original.media_type,
        )
