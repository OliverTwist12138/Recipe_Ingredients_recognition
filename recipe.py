import json
import torch
from PIL import Image
from torchvision import transforms

class Recommend():
    def __init__(self):
        self.ingredients = []
        self.recipe_all_contained = []
        self.recipe_partial_contained = []
        self.ingredients_to_recipe, self.recipe = self.load_recipe()
        model_path = './model/custom.pt'
        self.model = torch.load(model_path)

    def load_recipe(self):
        import pickle
        with open('ingredients_to_recipe.pickle', 'rb') as file:
            ingredients_to_recipe = pickle.load(file)
        with open('recipe.pickle', 'rb') as file:
            recipe = pickle.load(file)
        return ingredients_to_recipe, recipe

    def add_ingredients(self, input):
        self.ingredients.remove(name)
    def add_ingredients(self, input):
        import os
        if os.path.isfile(input):
            name = self.classify(input).lower()
        else:
            name = input
        self.ingredients.append(name)
        self.recipe_partial_contained = self.find_recipe_partial_contained()
        self.recipe_all_contained = self.find_recipe_all_contained()

    def find_recipe_partial_contained(self):
        keys = self.ingredients_to_recipe.keys()
        res = []
        for key in keys:
            for name in self.ingredients:
                if name in key:
                    res.extend(self.ingredients_to_recipe[key])
        return (list(set(res)))

    def find_recipe_all_contained(self):
        res = []
        for keys in self.recipe.keys():
            values = self.recipe[keys]
            if len(values[0]) <= len(self.ingredients):
                is_true = False
                # print(keys, values[0])
                for value in values[0]:
                    is_true = False
                    for ingredient in self.ingredients:
                        if ingredient in value:
                            is_true = True
                            break
                    if not is_true:
                        break
                if is_true:
                    res.append(keys)

        res = (list(set(res)))
        return res

    def show_recipe_partial_contained(self):
        print('Current ingredients:', self.ingredients)
        print('Following recipe(s) have part of the ingredients ready')
        self.print_recipe(self.recipe_partial_contained)

    def show_recipe_all_contained(self):
        print('Current ingredients:', self.ingredients)
        print('Following recipe(s) have all of the ingredients ready.')
        self.print_recipe(self.recipe_all_contained)

    def print_recipe(self, recipe):
        if len(recipe)==0:
            print('Sorry, corresponding recipe not found.')
        for i in recipe:
            print('Recipe Name: {},\nIngredients: {},\nDirections: {}\n'
                  .format(i, self.recipe[i][0], self.recipe[i][1]))

    def classify(self, img_path):
        new_fruit_names = {0: 'Apple', 1: 'Apricot', 2: 'Avocado', 3: 'Banana', 4: 'Beetroot', 5: 'Blueberry',
                           6: 'Cactus', 7: 'Cantaloupe',
                           8: 'Carambula', 9: 'Cauliflower', 10: 'Cherry', 11: 'Chestnut', 12: 'Clementine',
                           13: 'Cocos', 14: 'Corn',
                           15: 'Cucumber', 16: 'Dates', 17: 'Eggplant', 18: 'Fig', 19: 'Ginger', 20: 'Granadilla',
                           21: 'Grape',
                           22: 'Grapefruit', 23: 'Guava', 24: 'Hazelnut', 25: 'Huckleberry', 26: 'Persimmon',
                           27: 'Kiwi', 28: 'Kohlrabi',
                           29: 'Kumquats', 30: 'Lemon', 31: 'Lemon Meyer', 32: 'Limes', 33: 'Lychee', 34: 'Mandarine',
                           35: 'Mango',
                           36: 'Mango Red', 37: 'Mangosteen', 38: 'Passion fruit', 39: 'Melon Piel de Sapo',
                           40: 'Mulberry', 41: 'Nectarine',
                           42: 'Nut Forest', 43: 'Pecan Nut', 44: 'Onion', 45: 'Orange', 46: 'Papaya', 47: 'Passion',
                           48: 'Peach', 49: 'Pear',
                           50: 'Pepino', 51: 'Pepper', 52: 'Physalis', 53: 'Pineapple', 54: 'Pitahaya', 55: 'Plum',
                           56: 'Pomegranate',
                           57: 'Pomelo', 58: 'Potato', 59: 'Quince', 60: 'Rambutan', 61: 'Raspberry', 62: 'Redcurrant',
                           63: 'Salak',
                           64: 'Strawberry', 65: 'Tamarillo', 66: 'Tangelo', 67: 'Tomato', 68: 'Walnut',
                           69: 'Watermelon'}
        model = self.model
        model.eval()
        img = Image.open(img_path)
        tfms = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.6840, 0.5786, 0.5037], [0.3035, 0.3600, 0.3914])])
        img_tensor = tfms(img).to('cuda').unsqueeze(0)
        model.eval()  # turn the model to evaluate mode
        with torch.no_grad():  # does not calculate gradient
            class_index = model(img_tensor).argmax()
        return (new_fruit_names[int(class_index)])

def save_recipe(path):
    with open(path, 'r') as f:
        data = json.load(f)
    # json_keys = ['directions', 'fat', 'date', 'categories', 'calories', 'desc', 'protein', 'rating', 'title', '', 'sodium']
    recipe = {}
    ingredients_to_recipe = {}
    for line in data:
        try:
            if len(line['ingredients']) > 0:
                recipe[line['title']] = [line['ingredients'],line['directions'][0]]
                for ingredient in line['ingredients']:
                    try:
                        ingredients_to_recipe[ingredient].append(line['title'])
                    except:
                        ingredients_to_recipe[ingredient] = [line['title']]
        except:
            pass
    import pickle
    with open('ingredients_to_recipe.pickle', 'wb') as file:
        pickle.dump(ingredients_to_recipe, file)
    with open('recipe.pickle', 'wb') as file:
        pickle.dump(recipe, file)
    return ingredients_to_recipe, recipe



if __name__ == '__main__':
    # img_path = './fruits-360_dataset/fruits-360/Test/Tomato 3/2_100.jpg'
    # img_path = './test_imgs/result0.png'
    # save_recipe('./full_format_recipes.json')

    recommend_recipe = Recommend()
    recommend_recipe.add_ingredients('Hazelnut')
    # recommend_recipe.add_ingredients('rice')
    # # recommend_recipe.add_ingredients('beef')

    recommend_recipe.add_ingredients('./test_imgs/result2.png')
    recommend_recipe.show_recipe_all_contained()
    # recommend_recipe.show_recipe_partial_contained()