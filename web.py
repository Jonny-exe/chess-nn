#!/usr/bin/python3
from flask import Flask, request, redirect
import chess
import chess.svg
from game import Game

app = Flask(__name__)

@app.route("/", methods=["GET"])
def hello_world():
    if 'game' not in globals():
        global game
        game = Game(None, False)

    if request.args.get('reset', default=0, type=int) == 1:
        game = Game(None, False)

    svg = game.get_svg()
    game.play_best_move()
    return f"<html><p><img width=600 height=600 src='data:image/svg+xml;base64,{svg}'></img></p><br/>\
            <form action='/'><button type=submit>Hello</button></form> \
            <form action='/?reset=1'><input type=hidden name=reset value=1></input> <button type=submit>reset</button></form> \
            </html>"

if __name__ == "__main__":
    app.run(debug=True)

