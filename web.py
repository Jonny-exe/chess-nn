#!/usr/bin/python3
from flask import Flask, request
import chess
import chess.svg
from game import Game

game = Game(None, False)
app = Flask(__name__)
print(game)

@app.route("/")
def hello_world():
    svg = game.get_svg()
    game.play_best_move()
    print("HellO")
    return f"<html><p><img width=600 height=600 src='data:image/svg+xml;base64,{svg}'></img></p><br/>\
            <form action='/'><button type=submit>Hello</button></form> \
            </html>"

if __name__ == "__main__":
    app.run(debug=True)

