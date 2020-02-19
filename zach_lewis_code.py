import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spotipy
import spotipy.util as util
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN

SPOTIFY_CLIENT_ID = 'e3e67004e5664c2384c4f2df95e6d518'
SPOTIFY_CLIENT_SECRET = 'ceeb942aab2249b1ab2569ae8a6bee3f'
SPOTIFY_REDIRECT_URI = 'http://localhost:8888/callback/'
SPOTIFY_USER_ID = 'zmlewis2'

# Spotify authorization
scope = 'playlist-modify-public'
username = SPOTIFY_USER_ID
token = util.prompt_for_user_token(username, scope, SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI)
sp = spotipy.Spotify(auth=token)

# Define a list of ten of my favorite artists that I have recently listened to
artist_names = ['William Clark Green', 'Flatland Cavalry', 'Turnpike Troubadours', 'Kolby Cooper', 'Kody West',
                'Randy Rogers Band', 'Koe Wetzel', 'Parker McCollum', 'Read Southall Band', 'Jon Wolfe']

# Define an empty dictionary to place information on each of the ten artists in my list
artists = {}

# Loop through the list of artists to place artist id for each artist to the artists dictionary
for name in artist_names:
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']
    try:
        artists[items[0]['name']] = items[0]['id']
    except:
        print(name)

# Find out how many artists the search returned
print('Identified ' + str(len(artists)) + ' artists')  # Identified 10 artists

# Use the dictionary to get a list of artist ids
artist_ids = [v for v in artists.values()]

# Identify related artists for each of the ten favorite artists
new_artists = []
for artist_id in artist_ids:
    results = sp.artist_related_artists(artist_id)
    for i in range(1, len(results['artists'])):
        new_artist_id = results['artists'][i]['id']
        new_artists.append(new_artist_id)

# Remove duplicate artist ids in the list of new artists (there are 190 ids before removing duplicates)
new_artists = list(set(new_artists))  # 67 unique artists in the list

# Check the list and remove any artists that are in my original list of ten artists from the new artist list
new_artists = [x for x in new_artists if x not in artist_ids]  # This removed 8 ids from the list of new artists

# Now that I have lists of ids for my favorite artists and a list of ids for related artists, I want
# to create a playlist of songs from the new artists that I should like. To do this, I want to find songs from
# the new artists that are most like the top songs from the list of my favorite artists.

# To do this, I will get the top songs from the artists and create a data frame for the audio features of each song.
# Then, I will do the same for all the artists in the new artist list and combine the data frames. Then, I will use
# a KNN anomaly detection method in python through pyOD. This will basically group the songs and find a group of outlier
# songs. I will cut these songs out of the data and assume that all of the songs left are most similar to the group of
# songs from my favorite artists. Then, I will use this set of new songs that were found to be similar to the top
# songs of my favorite artists to create a playlist for listening to new music.

# First, get a list of top 10 tracks for my favorite artists
fav_tracks = []
for i in range(0, len(artist_ids)):
    uri = 'spotify:artist:' + artist_ids[i]
    results = sp.artist_top_tracks(uri)
    for track in results['tracks'][:10]:
        fav_tracks.append('spotify:track:' + track['id'])  # append the URIs of each artist's top tracks

print('Selected ' + str(len(fav_tracks)) + ' songs')  # Selected 100 songs

# Get a list of top 5 tracks for the new artists (want to keep the list of new songs small to make sure I have enough
# songs from favorite artists to cluster later)
new_tracks = []
for i in range(0, len(new_artists)):
    uri = 'spotify:artist:' + new_artists[i]
    results = sp.artist_top_tracks(uri)
    for track in results['tracks'][:5]:
        new_tracks.append('spotify:track:' + track['id'])  # append the URIs of each artist's top tracks

print('Selected ' + str(len(new_tracks)) + ' songs')  # Selected 295 songs

# Obtain audio features for the 100 top songs from my favorite artists
fav_features = pd.DataFrame()
for track_chunk in [fav_tracks[i:i + 20] for i in range(0, len(fav_tracks), 20)]:
    fav_features = fav_features.append(pd.DataFrame(sp.audio_features(track_chunk)))

# Obtain audio features for the 584 top songs from the related artists
new_features = pd.DataFrame()
for track_chunk in [new_tracks[i:i + 20] for i in range(0, len(new_tracks), 20)]:
    new_features = new_features.append(pd.DataFrame(sp.audio_features(track_chunk)))

# I want to add a column to both data frames indicating whether they belong to a favorite artist or a new artist
# NOTE: This column will later become an index so that it is not used in the KNN anomaly detection process
fav_features['type'] = 'fav'
new_features['type'] = 'new'

# Concatenate both data frames together for analysis
features = pd.concat((fav_features, new_features))

# Set the index to be track URI and type (I want to retain this information but not use it in analysis)
features.set_index(['uri', 'type'], inplace=True)

# Drop features in this data frame that I don't want to include in analysis (removing all character variables)
# I am also choosing to remove duration_ms because I do not care about track length
features.drop(['id', 'track_href', 'analysis_url', 'duration_ms'], axis=1, inplace=True)

# View the distribution of features for all 684 songs
features.describe()

# Standardize the features data frame
X = StandardScaler().fit_transform(features)

# Create and fit KNN algorithm for anomaly detection - contamination set at 1% but that number doesn't matter here
# since I will look at decision scores to decide which songs to include in the playlist
clf = KNN(contamination=0.1, n_neighbors=5)
clf.fit(X)

# Use the trained KNN model to provide decision scores for each observation
# NOTE: In this case, lower is better and high numbers indicate outliers
# I will use the decision scores to find 100 songs from new artists with the lowest decision scores. These songs will be
# most similar to the top songs by my favorite artists
features['decision_score'] = clf.decision_scores_

# Reset the index to identify songs by URI and type (fav/new)
features.reset_index(inplace=True)

# Subset the features to only contain songs by new artists
features = features[features['type'] == 'new']

# Sort observations by ascending decision score value and take only the top 100 songs to create my playlist
songs = features.nsmallest(100, 'decision_score')

# Create a data frame of new songs excluded from the playlist
excluded = features[~features.index.isin(songs.index)]

# How do the selected songs compare to the original favorite songs in terms of audio features?
# How do both of these distributions compare to the new songs that I excluded from my final list of songs?
# Compare the distributions for three variables I care the most about: acousticness, instrumentalness, and liveness
sns.distplot(songs['acousticness'], color='red', label='Acousticness for new songs chosen for playlist')
sns.distplot(excluded['acousticness'], color='blue', label='Acousticness for new songs excluded from playlist')
sns.distplot(fav_features['acousticness'], color='green', label='Acousticness for songs from my 10 favorite artists')
plt.legend()
# For acousticness, I notice a significant tail in the excluded distribution, while the distribution of selected songs
# closely follows the distribution of the original favorite songs. This is good for the analysis and means that the
# songs I have selected are similar overall in terms of acousticness to my favorite songs.

sns.distplot(songs['instrumentalness'], color='red', label='Acousticness for new songs chosen for playlist')
sns.distplot(excluded['instrumentalness'], color='blue', label='Acousticness for new songs excluded from playlist')
sns.distplot(fav_features['instrumentalness'], color='green', label='Acousticness for songs from 10 favorite artists')
plt.legend()
# This plot is hard to distinguish any distribution. Instrumentalness has really small values and may not have been
# significant in the KNN process.

sns.distplot(songs['liveness'], color='red', label='Acousticness for new songs chosen for playlist')
sns.distplot(excluded['liveness'], color='blue', label='Acousticness for new songs excluded from playlist')
sns.distplot(fav_features['liveness'], color='green', label='Acousticness for songs from 10 favorite artists')
plt.legend()
# Overall, it looks like the selected songs closely follows the distribution for my favorite songs. The distribution of
# excluded songs also follows closely as well but there is a significant portion of outliers in the right tail, which
# indicates that some outliers in terms of liveness were excluded from the selected playlist.

# This list of 100 songs will be used to create my new music playlist
songs = songs['uri']

# Create a playlist of the 100 new songs I identify from analysis
playlist_name = 'New Music for Zach Lewis'
playlist_description = 'Top 100 songs most similar to the best of my favorite Texas/Red Dirt Country artists'
playlist_json = sp.user_playlist_create(SPOTIFY_USER_ID, playlist_name)
for track_chunk in [songs[i:i + 20] for i in range(0, len(songs), 20)]:
    sp.user_playlist_add_tracks(SPOTIFY_USER_ID, playlist_id=playlist_json['id'], tracks=track_chunk)
