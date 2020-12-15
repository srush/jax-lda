import jax
import jax.numpy as np
from collections import Counter
c = Counter()

DOCS = 9500
VOCAB = 1000
TOKENS = 100
TOPICS = 25
DOCUMENT = "documents.txt"
STOPLIST = "stoplist.txt"

def bincount2d(arr, bins=None):
    if bins is None:
        bins = np.max(arr) + 1
    count = np.zeros(shape=[len(arr), bins], dtype=np.int64)
    indexing = (np.ones_like(arr).T * np.arange(len(arr))).T
    return count.at[indexing, arr].add(1)

# Process the document
stop = set([l.strip() for l in open(STOPLIST)])
stop |= set([".", "-",  "?"])
def parse(l):
    t = l.strip().split("\t")[-1].lower().replace("?", " ?").replace(".", " .").split()
    return [w for w in t if w not in stop]
data = []
for l in open(DOCUMENT):
    tokens = parse(l)
    for t in tokens:
        c[t] += 1
vocab = {v:i+1 for i, (v, _) in enumerate(c.most_common(VOCAB-1))}
vocab['pad'] = 0 # padding: 0

data = []
for l in open(DOCUMENT):
    tokens = parse(l)
    data.append([vocab.get(tokens[t], 0) if t < len(tokens) else 0 for t in range(TOKENS)])
data = data[:DOCS]

# Run LDA
key = jax.random.PRNGKey(0)
data = np.array(data)
topic_token = jax.random.randint(key, (DOCS, TOKENS), 1, TOPICS)

# Initialize token type 0 (paddings/stopwords) with topic 0
topic_token = topic_token.at[data == 0].set(0)

topic_document = bincount2d(topic_token)
topic_word = jax.ops.index_add(np.zeros((TOPICS, VOCAB)), 
                               jax.ops.index[topic_token.reshape(-1), data.reshape(-1)], 1)
topic_count = topic_word.sum(-1)

ALPHA, BETA = 0.1, 0.01
@jax.jit
def token_loop(state, scanned):
    topic_document, topic_word, topic_count = state
    topic_token, data, key = scanned

    topic_word = topic_word.at[topic_token, data].add(-1)
    topic_document = topic_document.at[topic_token].add(-1)
    topic_count = topic_count.at[topic_token].add(-1)

    # Resample
    dist = ((topic_word[:, data] + BETA) / (topic_count + VOCAB * BETA)) \
           * ((topic_document + ALPHA) / (TOKENS + TOPICS * ALPHA))
    new_topic = jax.random.categorical(key, np.log(dist / dist.sum()))

    topic_word = topic_word.at[new_topic, data].add(1)
    topic_document = topic_document.at[new_topic].add(1)
    topic_count = topic_count.at[new_topic].add(1)
    return (topic_document, topic_word, topic_count), (new_topic, data, key)

@jax.jit
def document_loop(state, scanned):
    # Indexed by doc 
    topic_word, = state
    topic_document, topic_token, data, key = scanned
    keys = jax.random.split(key, TOKENS)
    topic_count = topic_word.sum(-1)
    (topic_document, topic_word, topic_count), (topic_token, _, _) = \
        jax.lax.scan(token_loop, 
                     (topic_document, topic_word, topic_count), 
                     (topic_token, data, keys))   
    return (topic_word,), (topic_document, topic_token, data, key)

@jax.jit
def mcmc(i, state):
    topic_word, topic_document, topic_token, key = state
    keys = jax.random.split(key, DOCS + 1)

    (topic_word,), (topic_document, topic_token, _,  _)  = \
      jax.lax.scan(document_loop, 
                 (topic_word,),
                 (topic_document, topic_token, data, keys[1:])) 
    return topic_word, topic_document, topic_token, keys[0]

for i in range(50):
    (topic_word, topic_document, topic_token, key) = mcmc(i, (topic_word, topic_document, topic_token, key))
    print(i, topic_word[:, 24].sum())

rev = {v:k for k,v in vocab.items()}
out = topic_word / topic_word.sum(-1, keepdims=True)
for i in range(1, 21):
    print("TOPIC", i, [(rev[int(x)], float(out[i][x])) for x in reversed(np.argsort(out[i])[-10:])])
