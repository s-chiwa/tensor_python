import BaseHTTPServer
import cl_image
import cgi

#import base64
import os
import sys
import urllib
#import urlparse
#import binascii
#import struct

class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):

	def do_GET(self):
		body = b'Hello World'

		self.send_response(200)
		self.send_header('Content-type', 'text/html; charset=utf-8')
		self.send_header('Content-length', len(body))
		self.end_headers()
		self.wfile.write(body)

	def do_POST(self):
		
		rootPath = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),"test_data")
		import datetime
		today = datetime.datetime.today()
		dateRecord = today.strftime("%Y-%m-%d_%H-%M-%S")
		fileName = dateRecord + ".jpg"
		absoluteName = os.path.join(rootPath,fileName)

		#length = int(self.headers['Content-Length'])	
		#post_data = urlparse.parse_qs(self.rfile.read(length).decode('utf-8'))
		#print(post_data)
		form = cgi.FieldStorage(fp=self.rfile,headers=self.headers,environ={'REQUEST_METHOD':'POST','CONTENT_TYPE':self.headers['Content-Type'],})

		self.send_response(200)
		self.end_headers()

		if form.has_key("image_binary"):
			upFile = form["image_binary"]
			#print(upFile.value)
			with open(absoluteName, 'wb') as f:
				f.write(upFile.value)

			tnf = cl_image.TensorClassifyImage()
			rs = tnf.run_inference_on_image(absoluteName)

			self.wfile.write(rs)
		else:
			self.wfile.write(b"Analysis Data Nothing")


server_address = ('192.168.1.27', 8000)
httpd = BaseHTTPServer.HTTPServer(server_address, MyHandler)
httpd.serve_forever()
