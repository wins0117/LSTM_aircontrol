import pymongo
Myclient = pymongo.MongoClient("140.122.184.221",9487)
Myclient[‘class’].authenticate(‘IOT’,’IOT’) 
Mydb = Myclient[‘class’]
Listname = mydb.list_collection_names()
print(Listname)
