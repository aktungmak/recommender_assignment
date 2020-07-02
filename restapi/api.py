import csv
import sys
from bottle import route, run, template, abort

@route('/recommendations/<visitorid>')
def recommendations(visitorid):
    if visitorid in model:
        return {'visitorid': visitorid,
                'itemids':   model[visitorid]}
    else:
        abort(404, 'Invalid visitorid')

def load_model(filename):
    model = {}
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            model[row[0]] = row[1:]
    return model

if __name__ == '__main__':
    model = load_model(sys.argv[1])
    run(host='0.0.0.0', port=8080)
