# Get all of the Data
import os
import json
import wordninja

subfolders = [f.path for f in os.scandir(r"C:\Users\USER\Projects\tay-llm\Taylor-Swift-Lyrics\data\Albums") if f.is_dir() ]
data = []
for fold in subfolders:
    data.append([])
    for path in os.listdir(fold):
        if os.path.isfile(os.path.join(fold, path)):
            # We got the file boiz!
            file = open(os.path.join(fold, path), 'r', encoding="utf8")
            lines = file.readlines()
            lines.pop(0)
            lines.pop(len(lines) - 1)
            name_of_song = wordninja.split(path.removesuffix(".txt").replace("TaylorsVersion","").replace("_"," ").replace("Acoustic","").replace("AcousticVersion","").replace("PopVersion","").replace("LinerNotes","").replace("10MinuteVersion","").replace("originalversion","").replace("FromTheVault","").replace("Poem",""))
            string_1 = "Write a song in Taylor Swift's style called " + name_of_song + "->:!-> \n "
            string = string_1 + " ".join(lines)
            data.append({"prediction":string,"input":string_1})           

del data[0]
json.dump(data,open("prompts.json","w+"))