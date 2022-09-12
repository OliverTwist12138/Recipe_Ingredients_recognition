import pandas as pd
fruit_names = {0: 'Apple', 1: 'Apple', 2: 'Apple', 3: 'Apple', 4: 'Apple', 5: 'Apple', 6: 'Apple', 7: 'Apple', 8: 'Apple', 9: 'Apple', 10: 'Apple', 11: 'Apple', 12: 'Apple', 13: 'Apricot', 14: 'Avocado', 15: 'Avocado', 16: 'Banana', 17: 'Banana', 18: 'Banana', 19: 'Beetroot', 20: 'Blueberry', 21: 'Cactus', 22: 'Cantaloupe', 23: 'Cantaloupe', 24: 'Carambula', 25: 'Cauliflower', 26: 'Cherry', 27: 'Cherry', 28: 'Cherry', 29: 'Cherry', 30: 'Cherry', 31: 'Cherry', 32: 'Chestnut', 33: 'Clementine', 34: 'Cocos', 35: 'Corn', 36: 'Corn', 37: 'Cucumber', 38: 'Cucumber', 39: 'Dates', 40: 'Eggplant', 41: 'Fig', 42: 'Ginger', 43: 'Granadilla', 44: 'Grape', 45: 'Grape', 46: 'Grape', 47: 'Grape', 48: 'Grape', 49: 'Grape', 50: 'Grapefruit', 51: 'Grapefruit', 52: 'Guava', 53: 'Hazelnut', 54: 'Huckleberry', 55: 'Persimmon', 56: 'Kiwi', 57: 'Kohlrabi', 58: 'Kumquats', 59: 'Lemon', 60: 'Lemon Meyer', 61: 'Limes', 62: 'Lychee', 63: 'Mandarine', 64: 'Mango', 65: 'Mango Red', 66: 'Mangostan', 67: 'Maracuja', 68: 'Melon Piel de Sapo', 69: 'Mulberry', 70: 'Nectarine', 71: 'Nectarine', 72: 'Nut Forest', 73: 'Pecan Nut', 74: 'Onion', 75: 'Onion', 76: 'Onion', 77: 'Orange', 78: 'Papaya', 79: 'Passion', 80: 'Peach', 81: 'Peach', 82: 'Peach', 83: 'Pear', 84: 'Pear', 85: 'Pear', 86: 'Pear', 87: 'Pear', 88: 'Pear', 89: 'Pear', 90: 'Pear', 91: 'Pear', 92: 'Pepino', 93: 'Pepper', 94: 'Pepper', 95: 'Pepper', 96: 'Pepper', 97: 'Physalis', 98: 'Physalis', 99: 'Pineapple', 100: 'Pineapple', 101: 'Pitahaya', 102: 'Plum', 103: 'Plum', 104: 'Plum', 105: 'Pomegranate', 106: 'Pomelo', 107: 'Potato', 108: 'Potato', 109: 'Potato', 110: 'Potato', 111: 'Quince', 112: 'Rambutan', 113: 'Raspberry', 114: 'Redcurrant', 115: 'Salak', 116: 'Strawberry', 117: 'Strawberry', 118: 'Tamarillo', 119: 'Tangelo', 120: 'Tomato', 121: 'Tomato', 122: 'Tomato', 123: 'Tomato', 124: 'Tomato', 125: 'Tomato', 126: 'Tomato', 127: 'Tomato', 128: 'Tomato', 129: 'Walnut', 130: 'Watermelon'}
cnt = 0
match = {0:0}
for i in range(1,131):
    if fruit_names[i] != fruit_names[i-1]:
        cnt += 1
    match[i] = cnt


print(match)
import csv
csv_file = './fruit_test.csv'
with open(csv_file, "r") as infile, \
        open("fruit_test_cleaned.csv", "w", newline='') as outfile:
   reader = csv.reader(infile)
   next(reader, None)  # skip the headers
   writer = csv.writer(outfile)
   for row in reader:
       label = match[int(row[1])]
       writer.writerow([row[0],label])

