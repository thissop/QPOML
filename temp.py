import os 
base_path = './qpoml/tests/test_data/spectrum_CSVs/' 

for file in os.listdir(base_path): 
    lines = []
    file_path = base_path+file
    if file != '.gitkeep': 
            with open(file_path, 'r') as f: 
                    for line in f: 
                            line = line.replace('channel_','')
                            lines.append(line)
            with open(file_path, 'w') as f: 
                    for line in lines: 
                            f.write(line)
            if os.path.exists(file+'.csv'):
                os.remove(file_path+'.csv')