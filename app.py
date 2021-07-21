from flask import Flask,render_template,request
import caption_it

app=Flask(__name__)



@app.route('/')
def welcome():
    print('get')
    
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    if request.method=="POST":
        f=request.files['userfile']
        path='static/{}'.format(f.filename)
        print(path)
        f.save(path)
        caption=caption_it.caption_this(path)

        result_dict={'image':path,
                    'caption':caption}
      
        print (result_dict)

    return render_template('index.html',your_result=result_dict)






if __name__== '__main__':

    app.run()
