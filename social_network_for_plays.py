import pandas as pd
import numpy as np
from IPython.core.display import display, HTML 
import itertools
import networkx as nx
import sys
import matplotlib.pyplot as plt
from textblob import TextBlob
from gender import gender_by_name

sep = '||||'

# basic data wrangling and add new columns to be usde later
def prepare_df_for_plays(path):
    df = pd.read_csv(path)
    df = df.set_index('Dataline')

    df['ActSceneLine'] = df[['ActSceneLine']].applymap(lambda x: str(x))
    df['Act'] = [ np.nan if line == np.nan else line.split('.')[0] for line in df.ActSceneLine]
    df['Scene'] = [ np.nan if line == np.nan else line.split('.')[1] if len(line.split('.')) > 1 else np.nan for line in df.ActSceneLine]

    df = df.dropna()

    df['Act'] = [int(float(a)) for a in df['Act']]
    df['Scene'] = [int(float(s)) for s in df['Scene']]
    df['PlayerLinenumber'] = df[['PlayerLinenumber']].applymap(lambda x : round(x))
    df['Player'] = [p.title() for p in df['Player']]
    df['Play'] = [p.title() for p in df['Play']]
    df['Code'] = ''

    for i,r in df.iterrows():
        df.at[i, 'Code'] = r['Play'] + sep + '.'.join([str(r['Act']), str(r['Scene']), '{:03}'.format(r['PlayerLinenumber'])]) + sep + r['Player']

    return df

# returns a string that represents a relationship between two players
def make_relationship(a, b):
    return sep.join(sorted([a, b]))

# identifies conversation and participants, then encode them into new columns 
def prepare_df_by_encoding(df):
    m = df[['Code', 'PlayerLine']].groupby(['Code']).agg(lambda x: sep.join(x))
    m = m.reset_index()

    m['Play'] = [c.split(sep)[0] for c in m['Code']]
    m['Address'] = [c.split(sep)[1] for c in m['Code']]
    m['Act'] = [a.split('.')[0] for a in m['Address']]
    m['Scene'] = [a.split('.')[1] for a in m['Address']]
    m['Player'] = [c.split(sep)[2].title() for c in m['Code']]
    m['Player'] = [' '.join(p.split()) for p in m['Player']]

    m['Relationship'] = m['Target'] = ''
    
    for i, r in m.iterrows():
        line_number = int(r['Address'].split('.')[-1])
        is_first_line = line_number == 1
        is_last_line = i + 1 >= len(m) or m.at[i + 1, 'Act'] != r['Act'] or m.at[i + 1, 'Scene'] != r['Scene']
    
        # participants identification
        if not is_first_line and not is_last_line:
            prev = m.at[i - 1, 'Player']
            next = m.at[i + 1, 'Player']
            r['Target'] = prev if prev == next else ''
            r['Relationship'] = '' if r['Target'] == '' else  make_relationship(r['Player'], r['Target'])
    
    m = m.applymap(lambda x: x if x else np.NaN )
    return m

# returns a matrix that represents ralationships and relationship importance
def extract_matrix(df, play, encoded_df):    
    players = sorted(encoded_df[encoded_df.Play == play].Player.unique())
    players = [p for p in players if p not in ['All', 'Prologue']]
    
    d = {player: [0] * len(players) for player in players}
    
    matrix = pd.DataFrame(data=d, index=players)
        
    for relationship in itertools.combinations(players, 2):
        r = make_relationship(relationship[0], relationship[1])
        cnt = len(encoded_df[encoded_df.Relationship == r])
        matrix.at[relationship[1], relationship[0]] = matrix.at[relationship[0], relationship[1]] = cnt    
    
    return matrix    

# returns a float value that represent emotion statbility that calculated by standard deviations of sentiment values
def personality(df, play, player):    
    player_df = df[(df.Play == play) & (df.Player == player)]                          
    sentiments = [TextBlob(line).sentiment for line in player_df.PlayerLine]
    polarities = [abs(s.polarity) for s in sentiments]
    subjectivities = [s.subjectivity for s in sentiments]
    percentage = len(df[(df.Play == play) & (df.Player == player)]) / len(df[df.Play == play])
    
    score = round(np.std(polarities) * np.std(subjectivities) * 100 / (percentage + 1), 2)
    return score    

# returns something like '★★★' to represent an importance of a relationship
def importance_of_edge(edge, matrix):
    percentage = matrix.loc[edge[0], edge[1]]//(matrix.sum().sum()//100)
    return '★' * percentage

# network visualization
def plot_matrix(df, play, encoded_df):
    mat = extract_matrix(df, play, encoded_df)
    players = [c for c in list(mat.columns) if mat[c].sum() > 0]
    players = sorted(players, key=lambda p: personality(df=df, play=play, player=p))
        
    g = nx.Graph()
    g.add_nodes_from(players)
    
    for comb in itertools.combinations(players, 2):
        if mat.loc[comb[0], comb[1]] > 0:
            g.add_edge(comb[0], comb[1], weight=0.6)   
            
    pos = nx.spring_layout(g, k=1.9, scale=3)    
    
    _, __ = plt.subplots(1, 1, figsize=(24, 18))
    nx.draw_networkx_labels(g, pos, font_size=13, font_family='sans-serif')
        
    degrees = nx.degree(g)    
    cs = range(len(g.nodes))
    
    nx.draw(g, pos, nodelist=[d[0] for d in degrees], node_size=[d[1] * 1500 for d in degrees], node_color=cs, cmap=plt.cm.coolwarm, style='dotted')         
    nx.draw_networkx_edge_labels(g, pos, edge_labels={e : importance_of_edge(e, mat) for e in g.edges if(gender_by_name(e[0]) != gender_by_name(e[1]))}, font_color='r', font_size=12)
    nx.draw_networkx_edge_labels(g, pos, edge_labels={e : importance_of_edge(e, mat) for e in g.edges if(gender_by_name(e[0]) == gender_by_name(e[1]))}, font_color='b', font_size=12)
    
    plt.axis('off')
    plt.title("Social network from '{}'".format(play) , fontsize=24, color='orange')
    plt.show()