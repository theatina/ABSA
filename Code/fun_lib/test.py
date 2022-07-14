import csv

from attr import field

emb_dict =  { k:v for k,v in zip( ["lala", "lala2"], [1,2] ) }

csv_columns=["Sid","emb"]
csv_file = f"test_sEmbeddings_dict.csv"

with open(csv_file, 'w', newline='') as csv_file_w:  

    writer = csv.writer(csv_file_w, delimiter=',' )
    for key, value in emb_dict.items():
        writer.writerow([key, value])

with open(csv_file, mode='r') as infile:
    reader = csv.reader(infile, delimiter=',')
    mydict = {rows[0]:rows[1] for rows in reader}

print(mydict)