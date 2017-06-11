import ibmseti
import csv
import zipfile
import random

mydatafolder = '/Users/venkata.chintapalli/project/seti/explorers/test_data'
my_output_results = mydatafolder + '/signal_class_results.csv'

zz = zipfile.ZipFile(mydatafolder + '/' + 'basic4_test.zip')

def classify_model():
    return {'narrowband': random.uniform(0,1),
            'narrowbanddrd' : random.uniform(0,1),
            'noise' : random.uniform(0,1),
            'squiggle' : random.uniform(0,1)
           }
    

for fn in zz.namelist():
    data = zz.open(fn).read()
    aca = ibmseti.compamp.SimCompamp(data)
    uuid = aca.header()['uuid']
    cr = classify_model()
    
    #print(uuid)
    #spectrogram = draw_spectrogram(aca) #whatever signal processing code you need would go in your `draw_spectrogram` code

    #cr = class results. In this example, it's a dictionary. But in your experience it could be something else
    #       like a simple list.
    #cr = my_model.classify(spectrogram)

    with open(my_output_results, 'a') as csvfile:
        fwriter = csv.writer(csvfile, delimiter=',')
        fwriter.writerow([uuid, cr['narrowband'], cr['narrowbanddrd'], cr['noise'], cr['squiggle']])
