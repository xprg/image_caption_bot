from flask import Flask,redirect,render_template,request
from caption_it import caption_this

app=Flask(__name__)


@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    if request.method=="POST":
        f=request.files['userfile']
        path='static/{}'.format(f.filename)
        f.save(path)
        caption=caption_this(path)
        print(caption)
        print('its workin kinda')

    return render_template('index.html',caption=caption)






if __name__=='__main__':

    app.run(debug=True)