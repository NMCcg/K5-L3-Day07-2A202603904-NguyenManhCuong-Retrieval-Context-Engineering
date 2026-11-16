import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'starter'))
from ollama_controller import OllamaController


class Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.value).encode('utf-8')


class OllamaControllerTests(unittest.TestCase):
    def test_rejects_remote_host_to_keep_document_data_local(self):
        with self.assertRaises(ValueError):
            OllamaController(model='local-model', host='https://example.com')

    def test_native_search_tool_call_is_normalized(self):
        response = Response({'message': {
            'tool_calls': [{
                'function': {'name': 'search', 'arguments': {'query': 'backup path'}}
            }]
        }})
        with patch('ollama_controller.urlopen', return_value=response) as urlopen:
            controller = OllamaController(model='local-model')
            action = controller.decide('What happens after link failure?', [])
        self.assertEqual(action, {'action': 'search', 'query': 'backup path'})
        request = urlopen.call_args.args[0]
        self.assertIn('127.0.0.1', request.full_url)
        self.assertEqual(json.loads(request.data)['model'], 'local-model')


if __name__ == '__main__':
    unittest.main()
