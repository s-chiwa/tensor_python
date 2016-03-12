import BaseHTTPServer

class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):

	def do_GET(self):

		body = b'Hello World'

		self.send_response(200)
		self.send_header('Content-type', 'text/html; charset=utf-8')
		self.send_header('Content-length', len(body))
		self.end_headers()
		self.wfile.write(body)

server_address = ('192.168.0.9', 8000)
httpd = BaseHTTPServer.HTTPServer(server_address, MyHandler)
httpd.serve_forever()
