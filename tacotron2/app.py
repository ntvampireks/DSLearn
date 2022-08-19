from flask import Flask, render_template, Response, request, send_file
import sys
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
import tornado.ioloop
from tornado.ioloop import IOLoop
from Inference_tts import load_models, text_to_speak
import os

tacotron, waveglow = load_models("checkpoint_201000", "waveglow_6000")
app = Flask(__name__)

@app.route('/tts', methods=['POST'])
def texttospeech():
    if request.method == 'POST':
        result = request.form
        text = result['input_text']
        dict = result['input_dict']
        #max_duration_s = float(result['max_duration_s'])
        audio = text_to_speak(tacotron, waveglow, text, dict)
        return send_file('wavs\\' + audio, as_attachment=True)


@app.route('/')
def show_entries():
    return render_template('simple.html')


# Route to stream music

# launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    port = 31337
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.start()