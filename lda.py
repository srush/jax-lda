import jax
import jax.numpy as np
from collections import Counter
c = Counter()


VOCAB = 1000
TOPICS = 25
DOCUMENT = "documents.txt"
STOPLIST = "stoplist.txt"
EPOCHS = 50

# Preprocess the document
stop = set([l.strip() for l in open(STOPLIST)])
stop |= set([".", "-",  "?"])
def parse(l):
    t = l.strip().split("\t")[-1].lower().replace("?", " ?").replace(".", " .").split()
    return [w for w in t if w not in stop and len(w) > 4]
data = []
for l in open(DOCUMENT):
    tokens = parse(l)
    for t in tokens:
        c[t] += 1
vocab = {v:i+2 for i, (v, _) in enumerate(c.most_common(VOCAB))}

data = []
doc = []
for i, l in enumerate(open(DOCUMENT)):
    tokens = parse(l)
    data += [vocab.get(t, 0) for t in tokens]
    doc += [i] * len(tokens)
DOCUMENTS = i

SIZE = len(data)
print("Data Size", SIZE)
# Create tables
key = jax.random.PRNGKey(0)
data = np.array(data, dtype=np.int32)
docs = np.array(doc, dtype=np.int32)
topic_token = jax.random.randint(key, (SIZE,), 0, TOPICS, dtype=np.int32)

topic_word = jax.ops.index_add(np.zeros((TOPICS, VOCAB), dtype=np.int32), 
                               jax.ops.index[topic_token.reshape(-1), data.reshape(-1)], 1)
topic_document = jax.ops.index_add(np.zeros((DOCUMENTS, TOPICS), dtype=np.int32), 
                                   jax.ops.index[docs, topic_token], 1)
tokens_doc = np.bincount(docs, length=DOCUMENTS)


# Main code
ALPHA, BETA = 0.1, 0.01


def token_loop(state, scanned):
    topic_word, topic_document, topic_count = state
    topic_token, data, doc, key = scanned

    local_tw = topic_word[:, data].at[topic_token].add(-1)
    local_td = topic_document[doc].at[topic_token].add(-1)
    local_tc = topic_count.at[topic_token].add(-1)

    # Resample
    dist = ((local_tw + BETA) / (local_tc + VOCAB * BETA)) \
           * ((local_td + ALPHA) / (tokens_doc[doc] + TOPICS * ALPHA))
    new_topic = jax.random.categorical(key, np.log(dist))

    def update(_):
        return (topic_word.at[new_topic, data].add(1).at[topic_token, data].add(-1),
                topic_document.at[doc, new_topic].add(1).at[doc, topic_token].add(-1),
                topic_count.at[new_topic].add(1).at[topic_token].add(-1),
        )
    
    return jax.lax.cond((new_topic != topic_token),
                        update,
                        lambda _ : (topic_word, topic_document, topic_count), None
    ), (new_topic, None, None, None)    
    

@jax.jit
def mcmc(state):
    topic_count, topic_word, topic_document, topic_token, key = state
    keys = jax.random.split(key, SIZE + 1)
    (topic_word, topic_document, topic_count), (topic_token, _, _, _) = \
      jax.lax.scan(token_loop, 
                (topic_word, topic_document, topic_count), 
                (topic_token, data, docs, keys[1:]))
    return topic_count, topic_word, topic_document, topic_token, keys[0]

def run(topic_word, topic_document, topic_token):
    key = jax.random.PRNGKey(1)
    topic_count = topic_word.sum(-1)
    for i in range(EPOCHS):
        (topic_count, topic_word, topic_document, topic_token, key) =  \
          mcmc((topic_count, topic_word, topic_document, topic_token, key))
    return topic_word, topic_document, topic_token
topic_word, topic_document, topic_token = run(topic_word, topic_document, topic_token) 
        
# Print out
rev = {v:k for k,v in vocab.items()}
out = topic_word / topic_word.sum(0)
for i in range(TOPICS):
    print("TOPIC", i, [rev[int(x)] for x in reversed(np.argsort(out[i])[-5:]) if x > 1])
