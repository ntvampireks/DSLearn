from flask import Flask, render_template, request, send_file
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
import tornado.ioloop
from Inference_tts import load_models, text_to_speak

tacotron, waveglow = load_models("checkpoint_672500", "waveglow_6000") #нужны файлы с предобученными моделями
app = Flask(__name__)


@app.route('/tts', methods=['POST'])
def texttospeech():
    if request.method == 'POST':
        result = request.form
        text = result['input_text']
        dct = result['input_dict']
        audio = text_to_speak(tacotron, waveglow, text, dct)
        return send_file('wavs\\' + audio, as_attachment=True)


@app.route('/')
def show_entries():
    return render_template('simple.html')


if __name__ == "__main__":
    port = 31337
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.start()