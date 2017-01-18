import csv
def getIDS():
    with open("node_information.csv", "r") as f:
        reader = csv.reader(f)
        node_info = list(reader)

    with open("training_set.txt", "r") as f:
            reader = csv.reader(f)
            data_set = list(reader)

    data_set = [element[0].split(" ") for element in data_set]
    valid_ids=set()
    for element in data_set:
        valid_ids.add(element[0])
        valid_ids.add(element[1])
        
    if node_info is None:
        tmp=[element for element in node_info if element[0] in valid_ids ]
        node_info=tmp
        del tmp



    IDs = []
    ID_pos={}
    for element in node_info:
        ID_pos[element[0]]=len(IDs)
        IDs.append(element[0])
    return IDs
        