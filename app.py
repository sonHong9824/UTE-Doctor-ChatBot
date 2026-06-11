from __future__ import annotations

import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from chatbot import MedicalChatbot


assistant = MedicalChatbot()


class AssistantHandler(BaseHTTPRequestHandler):
    server_version = 'MedicalAssistantHTTP/1.0'

    def _set_headers(self, status_code: int, content_type: str = 'application/json; charset=utf-8') -> None:
        self.send_response(status_code)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _send_json(self, status_code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        try:
            self._set_headers(status_code)
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler interface
        self._set_headers(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler interface
        parsed = urlparse(self.path)
        path = parsed.path.rstrip('/') or '/'
        query = parse_qs(parsed.query)

        if path == '/health':
            self._send_json(HTTPStatus.OK, {
                'status': 'ok',
                'service': 'medical-assistant',
            })
            return

        if path == '/suggestions':
            mode = str(query.get('mode', ['general'])[0])
            self._send_json(HTTPStatus.OK, {
                'mode': mode,
                'suggestions': assistant.get_sample_prompts(mode),
            })
            return

        self._send_json(HTTPStatus.NOT_FOUND, {
            'error': 'Not found',
        })

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler interface
        path = urlparse(self.path).path.rstrip('/') or '/'
        if path != '/chat':
            self._send_json(HTTPStatus.NOT_FOUND, {
                'error': 'Not found',
            })
            return

        content_length = int(self.headers.get('Content-Length', '0'))
        raw_body = self.rfile.read(content_length or 0)

        try:
            payload = json.loads(raw_body.decode('utf-8') or '{}')
        except json.JSONDecodeError:
            self._send_json(HTTPStatus.BAD_REQUEST, {
                'error': 'Invalid JSON body',
            })
            return

        message = str(payload.get('message', '')).strip()
        mode = str(payload.get('mode', 'general'))
        history = payload.get('history', [])
        image_data_url = payload.get('imageDataUrl')
        image_file_name = payload.get('imageFileName')
        image_mime_type = payload.get('imageMimeType')

        if not message and not image_data_url:
            self._send_json(HTTPStatus.BAD_REQUEST, {
                'error': 'message or imageDataUrl is required',
            })
            return

        try:
            result = assistant.answer(
                message,
                history=history,
                mode=mode,
                image_data_url=image_data_url,
                image_file_name=image_file_name,
                image_mime_type=image_mime_type,
            )
            self._send_json(HTTPStatus.OK, result)
        except Exception as error:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {
                'reply': 'Hiện tại trợ lý AI đang tạm gián đoạn. Vui lòng thử lại sau ít phút.',
                'mode': mode,
                'source': 'fallback',
                'suggestions': assistant.get_sample_prompts(mode),
                'warning': str(error),
            })

    def log_message(self, format: str, *args) -> None:  # noqa: A003 - keep server logs quiet
        return


def main() -> None:
    port = int(os.getenv('PORT', '8765'))
    server = ThreadingHTTPServer(('0.0.0.0', port), AssistantHandler)
    print(f'Medical assistant service running on http://127.0.0.1:{port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
