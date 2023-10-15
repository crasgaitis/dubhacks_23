from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/select-genre-page.html')
def select_genre_page():
    return render_template('select-genre-page.html')

@app.route('/user-input-feelings-page.html')
def user_input_feelings_page():
    return render_template('user-input-feelings-page.html')

@app.route('/egg-page.html')
def egg_page():
    return render_template('egg-page.html')

@app.route('/loading-page.html')
def loading_page():
    return render_template('loading-page.html')

@app.route('/results.html')
def results_page():
    return render_template('results.html')


if __name__ == '__main__':
    app.run(debug=True)