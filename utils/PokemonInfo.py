import streamlit as sl
import requests
from PIL import Image
from io import BytesIO

# find pokemon avatar 
def get_pokemon_index(data):
    return data['id']

def get_pokemon_avatar(data):
    return data['sprites']['other']['official-artwork']['front_default']

# get pokemon element
def get_pokemon_element(data):
    return [data['types'][i]['type']['name'] for i in range(len(data['types']))]

# get pokemon evolution chain
def get_pokemon_evo_chain(data):
    # get evolution_chain json data
    try:
        evo_data = requests.get(requests.get(data['species']['url']).json()['evolution_chain']['url']).json()
        evo_name = []
        # get name of this pokemon based, first and second evolution
        evo_name.extend((evo_data['chain']['species']['name'],
                        evo_data['chain']['evolves_to'][0]['species']['name'],
                        *[evo_data['chain']['evolves_to'][0]['evolves_to'][i]['species']['name'] for i in range(len(evo_data['chain']['evolves_to'][0]['evolves_to']))]))
        return ' -> '.join(name for name in evo_name)
    
    except:
        return 'No evolution'

def get_pokemon_info(data):
    """Show"""
    sl.write(f'Pokedex number: #{get_pokemon_index(data)}')
    
    # show images
    img = Image.open(BytesIO(requests.get(get_pokemon_avatar(data)).content))
    sl.image(img, channels='BGR', width=350)
    
    # show element(s)
    sl.write('Element:', ', '.join(get_pokemon_element(data)))
    
    # show evolution chain
    sl.write('\nEvolution Chain:', get_pokemon_evo_chain(data))
